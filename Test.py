import glob
import sys
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations


SERIAL_PORT = None


def find_serial_port():
    """Find the USB-serial port for the Cerelog board across platforms."""
    if sys.platform == "darwin":
        # macOS: prefer /dev/cu.* over /dev/tty.* (tty.* blocks on open)
        candidates = glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.usbmodem*")
    elif sys.platform.startswith("linux"):
        candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    elif sys.platform.startswith("win"):
        # Windows: BrainFlow wants "COMx" — let pyserial enumerate them
        from serial.tools import list_ports
        candidates = [p.device for p in list_ports.comports()]
    else:
        candidates = []

    if not candidates:
        raise RuntimeError(
            "No USB-serial port found. Plug in the Cerelog board, or set "
            "SERIAL_PORT manually at the top of this file."
        )
    if len(candidates) > 1:
        print("Multiple serial ports found:", candidates)
        print("Using the first one. Set SERIAL_PORT manually to override.")
    return candidates[0]


port = SERIAL_PORT or find_serial_port()
print("Using serial port:", port)

BOARD_ID = BoardIds.CERELOG_X8_BOARD.value  # Cerelog ESP-EEG (ADS1299, 8ch)


BoardShim.enable_dev_board_logger()

params = BrainFlowInputParams()
params.serial_port = port
params.timeout = 15  # Cerelog firmware needs a moment to handshake

board = BoardShim(BOARD_ID, params)
board.prepare_session()
board.start_stream()

time.sleep(2)

sampling_rate = BoardShim.get_sampling_rate(BOARD_ID)
emg_channels = BoardShim.get_exg_channels(BOARD_ID)

print("Sampling rate:", sampling_rate)
print("EXG channels:", emg_channels)

# -----------------------------
# Electrode montage (bipolar EMG)
# -----------------------------
# emg_channels is the list of data-array rows for ports 1..8.
# emg_channels[0] = port 1, [1] = port 2, [2] = port 3, [3] = port 4 ...
# Each muscle = difference between its two electrode ports.
# The bias electrode is driven by the board hardware (not a data row).
BICEP    = (emg_channels[0], emg_channels[1])  # ports 1 & 2
SHOULDER = (emg_channels[2], emg_channels[3])  # ports 3 & 4

MUSCLES = [
    ("Bicep (ports 1-2)", BICEP),
    ("Shoulder (ports 3-4)", SHOULDER),
]
print("Bicep    = port1 - port2 -> rows", BICEP)
print("Shoulder = port3 - port4 -> rows", SHOULDER)



app = QtWidgets.QApplication([])

win = pg.GraphicsLayoutWidget(show=True, title="Cerelog ADS1299 EMG Test")

# One stacked plot per muscle. Each shows the filtered EMG (faint) plus a
# bright activation envelope that clearly rises when you flex.
curves = []
for i, (name, _pair) in enumerate(MUSCLES):
    plot = win.addPlot(row=i, col=0, title=name)
    plot.setLabel("left", "EMG (uV)")
    plot.setLabel("bottom", "Samples")
    plot.showGrid(x=False, y=True)
    raw_curve = plot.plot(pen=pg.mkPen((120, 120, 120), width=1))
    env_curve = plot.plot(pen=pg.mkPen("y", width=2))
    curves.append((raw_curve, env_curve))

WINDOW_SECONDS = 5
WINDOW_SAMPLES = sampling_rate * WINDOW_SECONDS

# ADS1299 full-scale input at gain 24 (4.5V ref): +/-187500 uV. Hitting this
# means the channel is railed (almost always a missing SRB1 reference).
RAIL_UV = 187500.0

def update():
    data = board.get_current_board_data(WINDOW_SAMPLES)

    if data.shape[1] < 32:  # need enough samples for the filters
        return

    # --- Saturation / reference check ---
    # Each channel reads CHx+ minus the SRB1 reference. If SRB1 isn't connected,
    # channels float to the ADS1299 rail (~+/-187500 uV at gain 24) and no muscle
    # signal gets through. Warn (throttled) so the hookup problem is obvious.
    update.frame = getattr(update, "frame", 0) + 1
    if update.frame % 30 == 0:  # ~once per second
        used_rows = sorted({r for _n, pair in MUSCLES for r in pair})
        railed = [r for r in used_rows
                  if np.max(np.abs(data[r] * 1e6)) > 0.95 * RAIL_UV]
        if railed:
            print(f"WARNING: channels at rows {railed} are SATURATED "
                  f"-> check the SRB1 reference electrode (separate from BIAS).")

    for (raw_curve, env_curve), (_name, (pos, neg)) in zip(curves, MUSCLES):
        # Bipolar EMG = difference of the muscle's two electrodes, in microvolts
        # (BrainFlow returns volts, so scale by 1e6). brainflow filters in place,
        # so the array must be contiguous float64.
        sig = np.ascontiguousarray((data[pos] - data[neg]) * 1e6)

        # Clean up: remove DC offset, notch out 60 Hz mains hum, keep EMG band.
        DataFilter.detrend(sig, DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandstop(sig, sampling_rate, 58, 62, 3, FilterTypes.BUTTERWORTH, 0)
        DataFilter.perform_highpass(sig, sampling_rate, 20.0, 4, FilterTypes.BUTTERWORTH, 0)

        # Activation envelope: rectify + smooth -> clear "muscle on" level.
        env = np.ascontiguousarray(np.abs(sig))
        DataFilter.perform_lowpass(env, sampling_rate, 5.0, 4, FilterTypes.BUTTERWORTH, 0)

        raw_curve.setData(sig)
        env_curve.setData(env)

timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)

try:
    app.exec()
finally:
    board.stop_stream()
    board.release_session()