# JustFan

A Windows and Linux desktop monitor for microphone noise, with placeholders for
temperature and fan-speed measurements. This version does not control fans or
execute sensor commands. All processing is local; audio is neither saved nor
transmitted.

## Requirements

- Python 3.11 or newer, with a desktop display available.
- An audio input device and microphone access.
- NumPy, PySide6, pyqtgraph, and sounddevice, installed from `requirements.txt`.

### Windows

Run these commands from the project directory in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe JustFan.py
```

The sounddevice Windows wheel includes PortAudio. Enable microphone access for
desktop applications in Windows privacy settings if capture is denied. No
administrator privileges are needed to run JustFan.

### Linux

Install PortAudio and the Qt display libraries with your distribution's package
manager. For Debian or Ubuntu:

```bash
sudo apt install python3-venv libportaudio2 libxcb-cursor0 libxkbcommon-x11-0
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python JustFan.py
```

Run inside a graphical desktop session. The input devices exposed by PortAudio
depend on the installed ALSA, PulseAudio, or PipeWire configuration. If Qt reports
a missing display plugin dependency, install the corresponding Qt/XCB or Wayland
runtime library for your distribution. Do not run the application as root.

## Input And Capture

JustFan selects the default input device and starts capture on launch. The input
selector includes each device's host API so that identically named devices can be
distinguished. Devices are sorted by name, then by host API, ignoring case.
Sorting preserves the default device and the current selection when refreshing.
Select another microphone or sound card to stop the previous stream, clear its
measurements, and rebuild the channel list and spectra.

The reported default sample rate, maximum input channel count, and low input
latency fill the capture controls automatically. The block/FFT size is chosen
from a power of two near 50 ms; PortAudio does not report a default FFT size.
Capture uses floating-point samples. Each channel corresponds to one input
channel, in driver order, rather than to a separate sound card.

Change the sample rate, channel count, block/FFT size, or requested latency and
press **Apply** to begin a new measurement. The driver validates the settings;
not every combination is supported. Requested latency is a hint and may differ
from actual driver latency. **Stop** freezes the display and releases the input;
**Start** begins a fresh measurement. **Reset peaks** retains current readings
while resetting peak levels, the held spectrum, and measured time. Held values
restart from the current readings, including when capture is stopped. The refresh
icon re-enumerates available devices and starts a new measurement on the selected
device.

Unsupported settings, missing devices, capture interruptions, and input
overflows are reported in the window. After reconnecting a device, refresh the
device list. A device exposed through multiple host APIs can have different
channel counts or supported settings for each API.

## Measurements And Layout

Drag the divider to resize the channel list and chart area continuously. Spectra
reflow into columns as space permits and scroll vertically for larger channel
counts. The left side shows three measurement rows per channel, with current
values aligned left and measurement peaks aligned right.

The bottom-right status bar shows **FPS** before the measured time. FPS counts
display updates with new audio data, averaged over one second, rather than timer
ticks or audio samples. It returns to zero when capture stops or no new frames
are displayed.

- **Noise (dBFS):** RMS amplitude over the most recent capture block. Peak is the
	highest block RMS level since capture started or peaks were reset, not the
	highest individual sample. Peaks are accumulated for every captured block,
	even when display updates skip intermediate blocks.
- **Temperature (C)** and **Speed (RPM):** `--` for current and peak values until
	sensor acquisition is implemented.
- **Noise spectrum:** a Hann-window FFT from zero to the Nyquist frequency,
	updated approximately 30 times per second when new audio is available. The
	amplitude is RMS dBFS per FFT bin with coherent-gain compensation; it is not
	a power spectral density. Frequency resolution is sample rate / FFT size.

Each plot overlays a solid **Current** spectrum and a dashed orange **Peak hold**
spectrum. Peak hold stores the highest amplitude reached independently at every
frequency bin since measurement start or the last reset, including audio blocks
between display refreshes. Peaks at different frequencies may come from different
times; this is not a frozen single-frame spectrum. Storage stays bounded to the
current and held spectra rather than an audio history.

The channel list's **Noise** peak is the highest whole-block RMS level, not the
largest FFT-bin amplitude; these two measurements normally have different values.
Every reported input channel has independent level and spectrum peaks. A driver's
channel count does not guarantee a distinct active microphone on every channel:
some routes may carry only silence or low-level noise.

The **Peak hold** checkbox beside the spectrum heading shows or hides the overlay
without discarding its history. **Reset peaks** resets both the noise-level peaks
and held spectra. Starting capture, applying settings, refreshing devices, or
changing devices starts new peak histories on the matching frequency grid.

The display floor is -120 dBFS. The plot extends 5 dB below that floor, and held
traces are drawn above current traces, so floor-level or overlapping peak traces
remain visible. The extra axis margin does not change the calculated values.
A full-scale sine wave has an RMS level near -3.01 dBFS. These are digital levels,
**not calibrated dB SPL**; microphone gain,
driver processing, and hardware all affect the readings. Measured time counts
captured samples and excludes stopped time; an input overflow can lose samples.

## Sensor Commands

Each channel has its own temperature and fan-speed commands. Open a channel's
configuration icon directly to edit that channel. The top-right **Sensor commands**
button opens an all-channel table, with one row per channel and separate columns
for temperature and fan-speed commands. All channels can be edited together
without switching a channel selector; the table scrolls for larger channel counts.
**Save** saves every row, while **Cancel** discards every pending edit.

The individual channel dialog remains available with its channel selector. Both
editors share the same saved configuration.

Commands are saved in per-user Qt settings by device name, host API, and channel
number, independently for each input device. Device index changes do not affect
these settings. Reducing the input channel count does not delete configurations
for the other channels; they return when those channels are enabled again.
Existing global commands are offered as Channel 01's initial values and migrate
to that device's Channel 01 on the first **Save**.

Window geometry and divider position are also saved in per-user Qt settings.
Commands are not executed or parsed in this version. Temperature and speed remain
unavailable regardless of the saved text.

## Tests

With the dependencies installed:

```bash
python -m unittest -v
```

The tests use generated signals, a mocked PortAudio stream, and Qt's offscreen
platform. They cover RMS and spectrum scaling, retained peaks, stream cleanup,
device changes, editable settings, splitter reflow, unavailable sensors, error
states, and saved command configuration. They do not access a microphone.
