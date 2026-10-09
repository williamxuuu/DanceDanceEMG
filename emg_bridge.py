"""
EMG bridge: Cerelog board  ->  Flex Tiles (browser).

The browser game cannot talk to the board directly (no serial access from a web
page). This script does what muscle_game.py does to READ the board, but instead
of running its own game it streams normalized muscle activation to the browser
over Server-Sent Events (SSE). The page subscribes and feeds every frame into
updateInputsFromEMG().

Pipeline (continuous, ~60 Hz):
    board -> sliding-window RMS per muscle -> ratio vs rest baseline
          -> normalized 0..1 -> JSON over http://localhost:8765/emg

Wiring (must match Test.py / muscle_game.py):
    Left Bicep    = port 1 - port 2   (exg[0] - exg[1])
    Left Shoulder = port 3 - port 4   (exg[2] - exg[3])
    Plus the SRB1 reference electrode AND the BIAS electrode, or the channels
    rail and nothing is detected.

Run:
    /Users/williamxu/.pyenv/versions/3.14.0/bin/python "emg_bridge.py"

Then open flex-tiles.html and click "Connect EMG" (it also auto-connects on load).
Keep this script running while you play.
"""

import glob
import json
import sys
import threading
import time

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# -----------------------------
# Config
# -----------------------------
SERIAL_PORT = None          # None = auto-detect the USB-serial port
HOST = "localhost"
PORT = 8765

WINDOW_SECONDS = 0.20       # sliding RMS window length (responsiveness vs noise)
STREAM_HZ = 60              # how often we push a frame to the browser
BASELINE_SECONDS = 3.0      # "relax" recording used to learn each muscle's rest

# Normalization: value = (ratio - 1) / (FLEX_RATIO - 1), clamped to 0..1, where
# ratio = current_level / rest_level. With FLEX_RATIO = 4, a 4x-rest flex maps to
# 1.0 and the game's default 0.6 threshold corresponds to ~2.8x rest (a clear
# flex). Tune the threshold live on the game's calibration screen.
FLEX_RATIO = 4.0
RAIL_UV = 187500.0          # ADS1299 full-scale at gain 24 -> railed channel

# Adaptive baseline: the resting EMG floor drifts over minutes (electrode
# settling), which otherwise makes a muscle creep past the threshold and stay
# "flexed". We slowly retrack the baseline ONLY when a muscle looks relaxed
# (ratio below the midpoint to a flex), so a real sustained flex never erodes it.
BASELINE_ADAPT = 0.004      # per-frame EMA factor (~4 s time constant at 60 Hz)
ADAPT_BELOW_RATIO = 1.0 + 0.5 * (FLEX_RATIO - 1.0)  # only adapt below this ratio

# Muscle id (must match the lane ids in flex-tiles.html) -> (pos exg idx, neg exg idx)
MUSCLE_PORTS = {
    "leftBicep":    (0, 1),   # ports 1 & 2
    "leftShoulder": (2, 3),   # ports 3 & 4
}

# Shared state written by the reader thread, read by the HTTP handlers.
latest = {m: 0.0 for m in MUSCLE_PORTS}
state_lock = threading.Lock()
running = True


def find_serial_port():
    if SERIAL_PORT:
        return SERIAL_PORT
    if sys.platform == "darwin":
        cands = glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.usbmodem*")
    elif sys.platform.startswith("linux"):
        cands = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    elif sys.platform.startswith("win"):
        from serial.tools import list_ports
        cands = [p.device for p in list_ports.comports()]
    else:
        cands = []
    if not cands:
        raise RuntimeError("No USB-serial port found. Plug in the board or set SERIAL_PORT.")
    return cands[0]


def connect():
    """Open the board, retrying the flaky handshake a few times."""
    params = BrainFlowInputParams()
    params.serial_port = find_serial_port()
    params.timeout = 15
    board = BoardShim(BoardIds.CERELOG_X8_BOARD.value, params)
    last = None
    for attempt in range(5):
        try:
            board.prepare_session()
            board.start_stream()
            return board
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  connect attempt {attempt + 1} failed, retrying... "
                  "(power-cycle the board and wait for the solid green LED)")
            time.sleep(4)
    raise RuntimeError(f"Could not connect to the board: {last}")


def activation_uv(sig, sampling_rate):
    """Filtered EMG -> RMS activation level in microvolts."""
    sig = np.ascontiguousarray(sig.astype(np.float64))
    DataFilter.detrend(sig, DetrendOperations.CONSTANT.value)
    DataFilter.perform_bandstop(sig, sampling_rate, 58, 62, 3, FilterTypes.BUTTERWORTH, 0)
    DataFilter.perform_highpass(sig, sampling_rate, 20.0, 4, FilterTypes.BUTTERWORTH, 0)
    return float(np.sqrt(np.mean(sig ** 2)))


def measure_levels(board, sampling_rate, exg, window_samples):
    """Read the most recent window. Returns ({muscle: RMS uV}, {muscle: railed})."""
    data = board.get_current_board_data(window_samples)  # does NOT consume samples
    if data.shape[1] < 8:           # not enough samples yet
        return {m: 0.0 for m in MUSCLE_PORTS}, {m: False for m in MUSCLE_PORTS}
    levels, railed = {}, {}
    for m, (pi, ni) in MUSCLE_PORTS.items():
        pos, neg = exg[pi], exg[ni]
        # Per-channel railing/saturation: a pinned (or pinning) electrode reads
        # garbage-high. We flag it so the stream can suppress it instead of
        # reporting a permanent "flex".
        r = (np.max(np.abs(data[pos] * 1e6)) > 0.95 * RAIL_UV or
             np.max(np.abs(data[neg] * 1e6)) > 0.95 * RAIL_UV)
        diff = (data[pos] - data[neg]) * 1e6
        levels[m] = activation_uv(diff, sampling_rate)
        railed[m] = bool(r)
    return levels, railed


def reader_thread():
    """Connect, calibrate a rest baseline, then stream normalized activation."""
    global running
    print("Connecting to Cerelog board...")
    board = connect()
    sampling_rate = BoardShim.get_sampling_rate(BoardIds.CERELOG_X8_BOARD.value)
    exg = BoardShim.get_exg_channels(BoardIds.CERELOG_X8_BOARD.value)
    window_samples = max(8, int(WINDOW_SECONDS * sampling_rate))

    # --- Calibrate: relax both muscles ---
    print(f"\n=== Calibrating: RELAX both muscles for {BASELINE_SECONDS:g} seconds ===")
    board.get_board_data()           # flush
    time.sleep(BASELINE_SECONDS)
    base_data = board.get_board_data()
    baseline = {}
    railed = False
    for m, (pi, ni) in MUSCLE_PORTS.items():
        pos, neg = exg[pi], exg[ni]
        railed = railed or np.max(np.abs(base_data[pos] * 1e6)) > 0.95 * RAIL_UV
        railed = railed or np.max(np.abs(base_data[neg] * 1e6)) > 0.95 * RAIL_UV
        diff = (base_data[pos] - base_data[neg]) * 1e6
        baseline[m] = max(activation_uv(diff, sampling_rate), 1.0)
    if railed:
        print("WARNING: channels SATURATED -> connect the SRB1 reference electrode "
              "(separate from BIAS). Detection won't work until then.")
    for m in MUSCLE_PORTS:
        print(f"  rest {m:13s} = {baseline[m]:7.1f} uV")
    print(f"\n=== Streaming on http://{HOST}:{PORT}/emg  (Ctrl+C to quit) ===\n")

    period = 1.0 / STREAM_HZ
    last_print = 0.0
    try:
        while running:
            t0 = time.time()
            levels, railed = measure_levels(board, sampling_rate, exg, window_samples)
            frame = {}
            for m in MUSCLE_PORTS:
                # 1) RAILING GUARD: a saturated channel is a bad electrode, not a
                #    flex. Report 0 so the game doesn't latch a permanent hit, and
                #    don't let the garbage value poison the baseline.
                if railed[m]:
                    frame[m] = 0.0
                    continue

                ratio = levels[m] / baseline[m]

                # 2) ADAPTIVE BASELINE: when the muscle looks relaxed, slowly
                #    retrack rest so electrode drift can't make it creep high and
                #    stay there. Skipped during a real flex (ratio is high).
                if ratio < ADAPT_BELOW_RATIO:
                    baseline[m] += BASELINE_ADAPT * (levels[m] - baseline[m])
                    baseline[m] = max(baseline[m], 1.0)

                v = (ratio - 1.0) / (FLEX_RATIO - 1.0)
                frame[m] = float(min(1.0, max(0.0, v)))
            with state_lock:
                latest.update(frame)

            # occasional console readout so you can sanity-check from the terminal.
            # Shows normalized value, raw uV, and per-muscle rail flag.
            if t0 - last_print > 0.5:
                last_print = t0
                parts = []
                for m in MUSCLE_PORTS:
                    tag = " RAILED" if railed[m] else ""
                    parts.append(f"{m}={frame[m]:.2f}({levels[m]:.0f}uV/base{baseline[m]:.0f}{tag})")
                print("\r  " + "   ".join(parts) + "        ", end="", flush=True)

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        board.stop_stream()
        board.release_session()
        print("\nBoard released.")


class EMGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence per-request logging

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") not in ("/emg", ""):
            self.send_response(404)
            self._cors()
            self.end_headers()
            return

        # Root path: tiny status page. /emg: SSE stream.
        if self.path.rstrip("/") == "":
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"EMG bridge is running. Stream at /emg")
            return

        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        period = 1.0 / STREAM_HZ
        try:
            while running:
                with state_lock:
                    payload = json.dumps(latest)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(period)
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser closed the tab / navigated away


def main():
    global running
    t = threading.Thread(target=reader_thread, daemon=True)
    t.start()
    server = ThreadingHTTPServer((HOST, PORT), EMGHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        running = False
        server.shutdown()
        # The reader is a daemon thread, so it would be killed mid-loop on exit.
        # Give it a moment to stop the stream and release the board cleanly.
        t.join(timeout=3)


if __name__ == "__main__":
    main()
