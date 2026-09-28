"""
EMG muscle game for the Cerelog ESP-EEG board.

Each round the game tells you a random muscle to flex. You flex it for ~3s,
and the game detects which muscle actually fired and scores you.

Wiring (matches Test.py):
    Bicep    = port 1 - port 2
    Shoulder = port 3 - port 4
    Plus an SRB1 reference electrode and a BIAS electrode (both required, or
    the channels rail and nothing is detected).

Run:
    /Users/williamxu/.pyenv/versions/3.14.0/bin/python "muscle_game.py"
"""

import glob
import sys
import time
import random

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

# -----------------------------
# Config
# -----------------------------
SERIAL_PORT = None          # None = auto-detect the USB-serial port
GET_READY_SECONDS = 2.0     # countdown before each flex
FLEX_SECONDS = 3.0          # how long you flex / how long we record
DETECT_RATIO = 2.5          # muscle must exceed 2.5x its rest level to count
RAIL_UV = 187500.0          # ADS1299 full-scale at gain 24 -> railed channel


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


def record_window(board, sampling_rate, channels, seconds):
    """Flush, record `seconds`, return {muscle: RMS uV} plus a railed flag."""
    board.get_board_data()              # flush old samples
    time.sleep(seconds)
    data = board.get_board_data()       # only the new window

    railed = False
    levels = {}
    for name, (pos, neg) in channels.items():
        railed = railed or np.max(np.abs(data[pos] * 1e6)) > 0.95 * RAIL_UV
        railed = railed or np.max(np.abs(data[neg] * 1e6)) > 0.95 * RAIL_UV
        diff = (data[pos] - data[neg]) * 1e6
        levels[name] = activation_uv(diff, sampling_rate)
    return levels, railed


def countdown(prefix, seconds):
    for s in range(int(seconds), 0, -1):
        print(f"\r  {prefix} {s}...", end="", flush=True)
        time.sleep(1)
    print("\r" + " " * 40, end="\r")


def main():
    print("Connecting to Cerelog board...")
    board = connect()
    sampling_rate = BoardShim.get_sampling_rate(BoardIds.CERELOG_X8_BOARD.value)
    exg = BoardShim.get_exg_channels(BoardIds.CERELOG_X8_BOARD.value)

    # Muscle -> (positive row, negative row).  Must match Test.py wiring.
    channels = {
        "BICEP":    (exg[0], exg[1]),   # ports 1 & 2
        "SHOULDER": (exg[2], exg[3]),   # ports 3 & 4
    }
    muscles = list(channels)

    print("\n=== Calibrating: RELAX both muscles ===")
    countdown("hold still", 3)
    baseline, railed = record_window(board, sampling_rate, channels, FLEX_SECONDS)
    if railed:
        print("WARNING: channels are SATURATED -> connect the SRB1 reference "
              "electrode (separate from BIAS). Detection won't work until then.")
    for m in muscles:
        print(f"  rest {m:9s} = {baseline[m]:7.1f} uV")
    # Guard against a zero baseline so ratios stay finite.
    baseline = {m: max(v, 1.0) for m, v in baseline.items()}

    print("\n=== GAME START ===  (Ctrl+C to quit)\n")
    rounds = 0
    correct = 0
    completed = 0   # rounds actually scored (excludes one interrupted by Ctrl+C)
    try:
        while True:
            rounds += 1
            target = random.choice(muscles)
            print(f"Round {rounds}:  >>> FLEX YOUR {target} <<<")
            countdown("get ready", GET_READY_SECONDS)
            print("  GO! flex now...", flush=True)

            levels, railed = record_window(board, sampling_rate, channels, FLEX_SECONDS)
            if railed:
                print("  (channels railed -> check SRB1 reference electrode)\n")
                rounds -= 1  # discarded round shouldn't count against the score
                continue

            ratios = {m: levels[m] / baseline[m] for m in muscles}
            best = max(muscles, key=lambda m: ratios[m])

            if ratios[best] < DETECT_RATIO:
                detected = "NONE (no clear flex)"
                hit = False
            else:
                detected = best
                hit = (best == target)

            completed += 1
            if hit:
                correct += 1
            mark = "CORRECT" if hit else "nope"
            detail = "  ".join(f"{m} {ratios[m]:.1f}x" for m in muscles)
            print(f"  -> detected: {detected:20s} [{mark}]   ({detail})")
            print(f"  score: {correct}/{completed}\n")
    except KeyboardInterrupt:
        print(f"\nFinal score: {correct}/{completed}")
    finally:
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()
