#!/usr/bin/env python3
"""Cross-platform microphone, temperature, and fan monitoring."""

import math
import signal
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock, Thread
from time import perf_counter
from urllib.parse import quote

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QByteArray, QEvent, QObject, QRectF, QSettings, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QColor, QFont, QFontDatabase, QIcon, QImage, QMouseEvent, QPainter, QPalette, QPixmap, QResizeEvent
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
	QAbstractItemView,
	QApplication,
	QCheckBox,
	QComboBox,
	QDialog,
	QDialogButtonBox,
	QDoubleSpinBox,
	QFormLayout,
	QFrame,
	QGridLayout,
	QHeaderView,
	QHBoxLayout,
	QLabel,
	QLineEdit,
	QMainWindow,
	QPushButton,
	QScrollArea,
	QSizePolicy,
	QSpinBox,
	QSplitter,
	QStackedWidget,
	QStyle,
	QTableWidget,
	QTableWidgetItem,
	QToolButton,
	QVBoxLayout,
	QWidget,
)

import pyqtgraph as pg


VERSION = "1.1"
DB_FLOOR = -120.0
DEFAULT_MAX_CHANNELS = 8
FFT_SIZES = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
CHANNEL_COLORS = ("#087f8c", "#c75b39", "#5364ab", "#ac4271", "#56842e", "#a87516")


def to_dbfs(amplitude: np.ndarray) -> np.ndarray:
	return 20.0 * np.log10(np.maximum(amplitude, 10.0 ** (DB_FLOOR / 20.0)))


class SpectrumAnalyzer:
	def __init__(self, sample_rate: float, size: int) -> None:
		if sample_rate <= 0 or size < 4:
			raise ValueError("A positive sample rate and at least four samples are required.")
		self.size = size
		self.frequencies = np.fft.rfftfreq(size, 1.0 / sample_rate)
		self.window = np.hanning(size)
		self.scale = np.sqrt(2.0) / self.window.sum()

	def analyze(self, samples: np.ndarray) -> np.ndarray:
		if samples.ndim != 2 or samples.shape[0] != self.size:
			raise ValueError("Audio must have shape (FFT size, channel count).")
		transform = np.fft.rfft(samples * self.window[:, None], axis=0)
		amplitudes = np.abs(transform) * self.scale
		amplitudes[0] /= np.sqrt(2.0)
		if self.size % 2 == 0:
			amplitudes[-1] /= np.sqrt(2.0)
		return to_dbfs(amplitudes)


@dataclass(frozen=True)
class AudioFrame:
	samples: np.ndarray
	levels: np.ndarray
	peaks: np.ndarray
	frames: int
	warning: str
	spectrum: np.ndarray | None = None
	spectrum_peaks: np.ndarray | None = None


class CaptureBuffer:
	"""Keep current readings and every block's level and spectrum peaks."""

	def __init__(self, channels: int, analyzer: SpectrumAnalyzer | None = None) -> None:
		self._lock = Lock()
		self._analyzer = analyzer
		self._samples: np.ndarray | None = None
		self._levels = np.full(channels, DB_FLOOR)
		self._peaks = np.full(channels, DB_FLOOR)
		self._spectrum: np.ndarray | None = None
		self._spectrum_peaks: np.ndarray | None = None
		self._frames = 0
		self._warning = ""

	def push(self, samples: np.ndarray, warning: str = "") -> None:
		samples = samples.copy()
		levels = to_dbfs(np.sqrt(np.mean(np.square(samples, dtype=np.float64), axis=0)))
		spectrum = self._analyzer.analyze(samples) if self._analyzer is not None else None
		with self._lock:
			self._samples = samples
			self._levels = levels
			self._peaks = np.maximum(self._peaks, levels)
			self._spectrum = spectrum
			if spectrum is not None:
				self._spectrum_peaks = spectrum if self._spectrum_peaks is None else np.maximum(self._spectrum_peaks, spectrum)
			self._frames += len(samples)
			if warning:
				self._warning = warning

	def snapshot(self) -> AudioFrame | None:
		with self._lock:
			if self._samples is None:
				return None
			return AudioFrame(
				self._samples,
				self._levels.copy(),
				self._peaks.copy(),
				self._frames,
				self._warning,
				self._spectrum.copy() if self._spectrum is not None else None,
				self._spectrum_peaks.copy() if self._spectrum_peaks is not None else None,
			)

	def reset_peaks(self) -> None:
		with self._lock:
			self._peaks = self._levels.copy()
			self._spectrum_peaks = self._spectrum.copy() if self._spectrum is not None else None
			self._frames = 0
			self._warning = ""


@dataclass(frozen=True)
class InputDevice:
	index: int
	name: str
	host_api: str
	channels: int
	sample_rate: float
	latency: float

	@property
	def label(self) -> str:
		return f"{self.name} [{self.host_api}]"


@dataclass(frozen=True)
class AudioSettings:
	sample_rate: float
	channels: int
	block_size: int
	latency: float


def query_input_devices() -> list[InputDevice]:
	host_apis = sd.query_hostapis()
	devices = []
	for index, info in enumerate(sd.query_devices()):
		if info["max_input_channels"] > 0:
			devices.append(
				InputDevice(
					index,
					str(info["name"]),
					str(host_apis[info["hostapi"]]["name"]),
					int(info["max_input_channels"]),
					float(info["default_samplerate"]),
					float(info["default_low_input_latency"]),
				)
			)
	return devices


class AudioCapture:
	STOP_TIMEOUT = 0.5

	def __init__(self) -> None:
		self.stream: sd.InputStream | None = None
		self.buffer: CaptureBuffer | None = None
		self.settings: AudioSettings | None = None

	def start(self, device: InputDevice, settings: AudioSettings) -> None:
		self.stop()
		self.buffer = None
		self.settings = None
		if not 1 <= settings.channels <= device.channels:
			raise ValueError("The channel count exceeds this device's input capabilities.")
		if settings.sample_rate <= 0 or settings.block_size not in FFT_SIZES:
			raise ValueError("Select a positive sample rate and a supported FFT size.")
		sd.check_input_settings(
			device=device.index,
			channels=settings.channels,
			samplerate=settings.sample_rate,
			dtype="float32",
		)
		buffer = CaptureBuffer(settings.channels, SpectrumAnalyzer(settings.sample_rate, settings.block_size))

		def callback(samples, _frames, _time_info, status) -> None:
			buffer.push(samples, str(status) if status else "")

		stream = sd.InputStream(
			device=device.index,
			samplerate=settings.sample_rate,
			channels=settings.channels,
			blocksize=settings.block_size,
			latency=settings.latency,
			dtype="float32",
			callback=callback,
		)
		try:
			stream.start()
		except Exception:
			stream.close()
			raise
		self.stream = stream
		self.buffer = buffer
		self.settings = settings

	def stop(self) -> None:
		stream, self.stream = self.stream, None
		if stream is not None:
			def close_stream() -> None:
				try:
					try:
						stream.stop()
					finally:
						stream.close()
				except sd.PortAudioError:
					pass

			thread = Thread(target=close_stream, daemon=True)
			thread.start()
			thread.join(self.STOP_TIMEOUT)


class FanIconAnimator(QObject):
	POSE_FRAMES = 8
	SPIN_FRAMES = 12
	SIZES = (16, 24, 32, 48, 64, 128, 256)
	IDLE_TRANSFORM = (0.9, -0.15, 0.06, 0.92, 0.0, 34.0)
	RUNNING_TRANSFORM = (0.92, 0.0, 0.0, 0.92, 4.8, 16.0)

	def __init__(self, window: QMainWindow) -> None:
		super().__init__(window)
		self._window = window
		self.idle_icon = QIcon(str(Path(__file__).with_name("JustFan.ico")))
		self.frames = self._load_frames(Path(__file__).with_name("JustFan.svg"))
		self.frame_index = 0
		self.timer = QTimer(self)
		self.timer.setInterval(50)
		self.timer.setTimerType(Qt.TimerType.PreciseTimer)
		self.timer.timeout.connect(self._advance)

	@classmethod
	@lru_cache(maxsize=1)
	def _load_frames(cls, path: Path) -> tuple[QIcon, ...]:
		try:
			root = ET.parse(path).getroot()
		except (OSError, ET.ParseError):
			return ()
		assembly = next((element for element in root.iter() if element.get("id") == "fanAssembly"), None)
		blades = next((element for element in root.iter() if element.get("id") == "fanBlades"), None)
		if assembly is None or blades is None:
			return ()
		ET.register_namespace("", "http://www.w3.org/2000/svg")
		ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
		frames = []
		for frame in range(cls.POSE_FRAMES + cls.SPIN_FRAMES):
			progress = min(frame, cls.POSE_FRAMES - 1) / (cls.POSE_FRAMES - 1)
			progress = progress * progress * (3.0 - 2.0 * progress)
			transform = [start + (end - start) * progress for start, end in zip(cls.IDLE_TRANSFORM, cls.RUNNING_TRANSFORM)]
			assembly.set("transform", f"matrix({' '.join(f'{value:.6g}' for value in transform)})")
			angle = max(0, frame - cls.POSE_FRAMES) * 360.0 / (7 * cls.SPIN_FRAMES)
			blades.set("transform", f"rotate({angle:.6g} 120 120)")
			renderer = QSvgRenderer(QByteArray(ET.tostring(root, encoding="utf-8")))
			if not renderer.isValid():
				return ()
			icon = QIcon()
			for size in cls.SIZES:
				image = QImage(size, size, QImage.Format.Format_ARGB32)
				image.fill(Qt.GlobalColor.transparent)
				painter = QPainter(image)
				try:
					painter.setRenderHint(QPainter.RenderHint.Antialiasing)
					renderer.render(painter, QRectF(0, 0, size, size))
				finally:
					painter.end()
				icon.addPixmap(QPixmap.fromImage(image))
			frames.append(icon)
		return tuple(frames)

	def _show_icon(self, icon: QIcon) -> None:
		QApplication.setWindowIcon(icon)
		self._window.setWindowIcon(icon)

	def set_running(self, running: bool) -> None:
		if running and self.frames:
			if not self.timer.isActive():
				self.frame_index = 0
				self._show_icon(self.frames[0])
				self.timer.start()
		else:
			self.timer.stop()
			self.frame_index = 0
			self._show_icon(self.idle_icon)

	def _advance(self) -> None:
		if not self.timer.isActive() or not self.frames:
			return
		self.frame_index += 1
		if self.frame_index >= len(self.frames):
			self.frame_index = self.POSE_FRAMES
		self._show_icon(self.frames[self.frame_index])


class ChannelWidget(QFrame):
	sensor_commands_requested = Signal(int)
	enabled_changed = Signal(int, bool)

	def __init__(self, channel: int, parent: QWidget | None = None) -> None:
		super().__init__(parent)
		self._channel = channel
		self.setObjectName("channel")
		self.setMinimumWidth(264)
		layout = QGridLayout(self)
		layout.setContentsMargins(12, 12, 12, 12)
		layout.setHorizontalSpacing(12)
		layout.setVerticalSpacing(10)
		self.enabled_checkbox = QCheckBox(f"Channel {channel + 1:02d}")
		self.enabled_checkbox.setChecked(True)
		self.enabled_checkbox.setStyleSheet(f"color: {CHANNEL_COLORS[channel % len(CHANNEL_COLORS)]}; font-weight: 600;")
		self.enabled_checkbox.setToolTip("Show this channel in the noise spectrum.")
		self.enabled_checkbox.toggled.connect(self._request_enabled_change)
		self.commands_button = QToolButton()
		self.commands_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
		self.commands_button.setToolTip(f"Configure Channel {channel + 1:02d} sensor commands")
		self.commands_button.setAccessibleName(f"Configure Channel {channel + 1:02d} sensor commands")
		self.commands_button.clicked.connect(self._request_sensor_commands)
		header = QHBoxLayout()
		header.addWidget(self.enabled_checkbox)
		header.addStretch()
		header.addWidget(self.commands_button)
		layout.addLayout(header, 0, 0)
		layout.addWidget(QLabel("Current"), 0, 1, Qt.AlignmentFlag.AlignLeft)
		layout.addWidget(QLabel("Peak"), 0, 2, Qt.AlignmentFlag.AlignRight)
		self.current_values = []
		self.peak_values = []
		for row, name in enumerate(("Noise (dBFS)", "Temperature (C)", "Speed (RPM)"), 1):
			layout.addWidget(QLabel(name), row, 0)
			current = QLabel("--")
			peak = QLabel("--")
			for label in (current, peak):
				label.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
				label.setMinimumWidth(max(46, label.fontMetrics().horizontalAdvance(f"{DB_FLOOR:.1f}") + 4))
			current.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
			peak.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
			layout.addWidget(current, row, 1)
			layout.addWidget(peak, row, 2)
			self.current_values.append(current)
			self.peak_values.append(peak)
		layout.setColumnStretch(1, 1)
		layout.setColumnStretch(2, 1)
		self.current_values[0].setToolTip("Block RMS level in dBFS; not calibrated sound pressure (dB SPL).")
		self.peak_values[0].setToolTip("Highest block RMS level in this measurement, not sample-peak amplitude.")
		for label in self.current_values[1:] + self.peak_values[1:]:
			label.setToolTip("Sensor command execution is not available in this version.")

	def _request_sensor_commands(self) -> None:
		self.sensor_commands_requested.emit(self._channel)

	def _request_enabled_change(self, enabled: bool) -> None:
		self.enabled_changed.emit(self._channel, enabled)

	def update_levels(self, current: float, peak: float) -> None:
		self.current_values[0].setText(f"{current:.1f}")
		self.peak_values[0].setText(f"{peak:.1f}")


class SpectrumPanel(QScrollArea):
	def __init__(self, parent: QWidget | None = None) -> None:
		super().__init__(parent)
		self.plots: list[pg.PlotWidget] = []
		self.curves = []
		self.peak_curves = []
		self._peak_hold_visible = True
		self._enabled_channels: list[bool] = []
		self._focused_plot: pg.PlotWidget | None = None
		self.columns = 0
		self._empty: QLabel | None = None
		self.setWidgetResizable(True)
		self.setFrameShape(QFrame.Shape.NoFrame)
		self.content = QWidget()
		self.grid = QGridLayout(self.content)
		self.grid.setContentsMargins(12, 8, 12, 12)
		self.grid.setSpacing(12)
		self.setWidget(self.content)

	def set_channels(self, count: int, sample_rate: float) -> None:
		self._focused_plot = None
		for plot in self.plots:
			self.grid.removeWidget(plot)
			plot.hide()
			plot.deleteLater()
		while self.grid.count():
			item = self.grid.takeAt(0)
			widget = item.widget()
			if widget is not None:
				widget.deleteLater()
		self.plots.clear()
		self.curves.clear()
		self.peak_curves.clear()
		self._enabled_channels = [True] * count
		self.columns = 0
		self._empty = None
		if not count:
			self._empty = QLabel("No spectrum data")
			self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
			self.grid.addWidget(self._empty, 0, 0)
			return
		for channel in range(count):
			plot = pg.PlotWidget(background="w")
			plot.viewport().installEventFilter(self)
			plot.setMinimumSize(300, 250)
			plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
			color = CHANNEL_COLORS[channel % len(CHANNEL_COLORS)]
			plot.setTitle(f"Channel {channel + 1:02d}", color=color, size="11pt")
			plot.setLabel("bottom", "Frequency", units="Hz")
			plot.setLabel("left", "RMS amplitude", units="dBFS")
			plot.showGrid(x=True, y=True, alpha=0.15)
			plot.setXRange(0, sample_rate / 2, padding=0)
			plot.setYRange(DB_FLOOR - 5, 0, padding=0)
			plot.setLimits(xMin=0, xMax=sample_rate / 2, yMin=DB_FLOOR - 10, yMax=12)
			plot.hideButtons()
			plot.addLegend(offset=(-10, 10), labelTextSize="9pt")
			curve = plot.plot(pen=pg.mkPen(color, width=0), name="Current", antialias=False)
			curve.setZValue(1)
			peak_curve = plot.plot(
				pen=pg.mkPen("#b45f16", width=1.0, style=Qt.PenStyle.DashLine),
				name="Peak hold" if self._peak_hold_visible else None,
				antialias=False,
			)
			peak_curve.setZValue(2)
			for trace in (curve, peak_curve):
				trace.setDownsampling(auto=True, method="peak")
				trace.setClipToView(True)
			peak_curve.setVisible(self._peak_hold_visible)
			self.plots.append(plot)
			self.curves.append(curve)
			self.peak_curves.append(peak_curve)
		self._reflow()

	def set_channel_enabled(self, channel: int, enabled: bool) -> None:
		if not 0 <= channel < len(self.plots) or self._enabled_channels[channel] == enabled:
			return
		self._enabled_channels[channel] = enabled
		if not enabled and self._focused_plot is self.plots[channel]:
			self._focused_plot = None
		self.columns = 0
		self._reflow()

	def set_peak_hold_visible(self, visible: bool) -> None:
		if visible == self._peak_hold_visible:
			return
		self._peak_hold_visible = visible
		for plot, curve in zip(self.plots, self.peak_curves):
			curve.setVisible(visible)
			if visible:
				plot.plotItem.legend.addItem(curve, "Peak hold")
			else:
				plot.plotItem.legend.removeItem(curve)

	def eventFilter(self, watched: QObject, event: QEvent) -> bool:
		if (
			event.type() == QEvent.Type.MouseButtonDblClick
			and isinstance(event, QMouseEvent)
			and event.button() == Qt.MouseButton.LeftButton
		):
			for channel, plot in enumerate(self.plots):
				if watched is plot.viewport() and self._enabled_channels[channel]:
					self._focused_plot = None if self._focused_plot is plot else plot
					self.columns = 0
					self._reflow()
					return True
		return super().eventFilter(watched, event)

	def _reflow(self) -> None:
		if not self.plots:
			return
		enabled_plots = [plot for plot, enabled in zip(self.plots, self._enabled_channels) if enabled]
		plots = [self._focused_plot] if self._focused_plot is not None else enabled_plots
		available = max(1, self.viewport().width() - 24)
		columns = 1 if self._focused_plot is not None else max(1, min(len(plots), (available + 12) // 372))
		if columns == self.columns:
			return
		while self.grid.count():
			self.grid.takeAt(0)
		if self._empty is not None:
			self._empty.hide()
			self._empty.deleteLater()
			self._empty = None
		for column in range(self.grid.columnCount()):
			self.grid.setColumnStretch(column, 0)
		for row in range(self.grid.rowCount()):
			self.grid.setRowStretch(row, 0)
		for index, plot in enumerate(plots):
			self.grid.addWidget(plot, index // columns, index % columns)
		if not plots:
			self._empty = QLabel("No channels enabled")
			self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
			self.grid.addWidget(self._empty, 0, 0)
		for channel, plot in enumerate(self.plots):
			plot.setVisible(self._enabled_channels[channel] and (self._focused_plot is None or plot is self._focused_plot))
		for column in range(columns):
			self.grid.setColumnStretch(column, 1)
		self.grid.setRowStretch(0 if self._focused_plot is not None else math.ceil(len(plots) / columns), 1)
		self.columns = columns

	def resizeEvent(self, event: QResizeEvent) -> None:
		super().resizeEvent(event)
		self._reflow()

	def update_spectra(self, frequencies: np.ndarray, levels: np.ndarray, peaks: np.ndarray) -> None:
		for channel, curve in enumerate(self.curves):
			curve.setData(frequencies, levels[:, channel])
			self.peak_curves[channel].setData(frequencies, peaks[:, channel])


class MainWindow(QMainWindow):
	def __init__(self, settings: QSettings | None = None) -> None:
		super().__init__()
		self.settings = settings if settings is not None else QSettings("JustFan", "JustFan")
		self.capture = AudioCapture()
		self.icon_animator = FanIconAnimator(self)
		self.analyzer: SpectrumAnalyzer | None = None
		self.channel_widgets: list[ChannelWidget] = []
		self._disabled_channels: dict[tuple[str, str], set[int]] = {}
		self._last_frames = -1
		self._last_warning = ""
		self._fps_frames = 0
		self._fps_started = perf_counter()
		self.setWindowTitle(f"JustFan v{VERSION}")
		self.resize(1360, 820)
		self.setMinimumSize(860, 560)
		self._build_ui()
		geometry = self.settings.value("window/geometry")
		splitter_state = self.settings.value("window/splitter")
		if isinstance(geometry, QByteArray):
			self.restoreGeometry(geometry)
		if isinstance(splitter_state, QByteArray):
			self.splitter.restoreState(splitter_state)
		self.timer = QTimer(self)
		self.timer.setInterval(33)
		self.timer.timeout.connect(self._update_measurement)
		self.timer.start()
		self.refresh_devices()

	def _build_ui(self) -> None:
		root = QWidget()
		self.setCentralWidget(root)
		layout = QVBoxLayout(root)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setSpacing(0)
		controls = QWidget()
		controls.setObjectName("controls")
		control_layout = QVBoxLayout(controls)
		control_layout.setContentsMargins(18, 12, 18, 14)
		control_layout.setSpacing(12)
		device_row = QHBoxLayout()
		device_row.addWidget(QLabel("Input device"))
		self.device_combo = QComboBox()
		self.device_combo.setMinimumContentsLength(16)
		self.device_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
		self.device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
		self.device_combo.currentIndexChanged.connect(self._select_device)
		device_row.addWidget(self.device_combo, 1)
		refresh = QToolButton()
		refresh.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
		refresh.setToolTip("Refresh input devices")
		refresh.setAccessibleName("Refresh input devices")
		refresh.clicked.connect(self.refresh_devices)
		device_row.addWidget(refresh)
		self.start_button = QPushButton("Start")
		self.start_button.clicked.connect(self._toggle_capture)
		device_row.addWidget(self.start_button)
		self.reset_button = QPushButton("Reset peaks")
		self.reset_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton))
		self.reset_button.setToolTip("Reset level and spectrum peaks and measured time; retain current readings.")
		self.reset_button.clicked.connect(self._reset_measurement)
		device_row.addWidget(self.reset_button)
		control_layout.addLayout(device_row)
		parameters = QGridLayout()
		parameters.setHorizontalSpacing(12)
		self.sample_rate = QDoubleSpinBox()
		self.sample_rate.setRange(1000, 768000)
		self.sample_rate.setDecimals(1)
		self.sample_rate.setSingleStep(1000)
		self.sample_rate.setSuffix(" Hz")
		self.channel_count = QSpinBox()
		self.channel_count.setRange(1, 1)
		self.fft_size = QComboBox()
		for size in FFT_SIZES:
			self.fft_size.addItem(f"{size:,} samples", size)
		self.fft_size.setToolTip("Capture block and Hann-window FFT length.")
		self.latency = QDoubleSpinBox()
		self.latency.setRange(0.1, 10000)
		self.latency.setDecimals(1)
		self.latency.setSuffix(" ms")
		self.latency.setToolTip("Requested input latency; the audio driver may choose a different value.")
		fields = (
			("Sample rate", self.sample_rate),
			("Input channels", self.channel_count),
			("Block / FFT size", self.fft_size),
			("Input latency", self.latency),
		)
		for column, (label, widget) in enumerate(fields):
			parameters.addWidget(QLabel(label), 0, column)
			parameters.addWidget(widget, 1, column)
			parameters.setColumnStretch(column, 1)
		self.commands_button = QPushButton("Sensor commands")
		self.commands_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
		self.commands_button.clicked.connect(self._configure_all_commands)
		self.commands_button.setMinimumWidth(self.commands_button.sizeHint().width())
		parameters.addWidget(self.commands_button, 0, 4, Qt.AlignmentFlag.AlignRight)
		self.apply_button = QPushButton("Apply")
		self.apply_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
		self.apply_button.setToolTip("Apply capture settings and start a new measurement.")
		self.apply_button.clicked.connect(self.start_capture)
		parameters.addWidget(self.apply_button, 1, 4, Qt.AlignmentFlag.AlignRight)
		control_layout.addLayout(parameters)
		layout.addWidget(controls)
		self.setMinimumWidth(max(self.minimumWidth(), controls.minimumSizeHint().width()))
		self.notice = QLabel()
		self.notice.setObjectName("notice")
		self.notice.setWordWrap(True)
		self.notice.setTextFormat(Qt.TextFormat.PlainText)
		self.notice.setContentsMargins(18, 8, 18, 8)
		self.notice.hide()
		layout.addWidget(self.notice)
		self.splitter = QSplitter(Qt.Orientation.Horizontal)
		self.splitter.setChildrenCollapsible(False)
		self.splitter.setOpaqueResize(True)
		self.splitter.setHandleWidth(7)
		left = QWidget()
		left.setMinimumWidth(292)
		left_layout = QVBoxLayout(left)
		left_layout.setContentsMargins(0, 0, 0, 0)
		left_heading = QHBoxLayout()
		left_heading.setContentsMargins(12, 12, 12, 0)
		left_heading.addWidget(QLabel("Channels"))
		left_heading.addStretch()
		self.channel_label = QLabel("0")
		left_heading.addWidget(self.channel_label)
		left_layout.addLayout(left_heading)
		channel_scroll = QScrollArea()
		channel_scroll.setWidgetResizable(True)
		channel_scroll.setFrameShape(QFrame.Shape.NoFrame)
		channel_content = QWidget()
		self.channel_layout = QVBoxLayout(channel_content)
		self.channel_layout.setContentsMargins(12, 8, 12, 12)
		self.channel_layout.setSpacing(10)
		channel_scroll.setWidget(channel_content)
		left_layout.addWidget(channel_scroll)
		right = QWidget()
		right.setMinimumWidth(346)
		right_layout = QVBoxLayout(right)
		right_layout.setContentsMargins(0, 0, 0, 0)
		chart_heading = QHBoxLayout()
		chart_heading.setContentsMargins(12, 12, 12, 0)
		chart_heading.addWidget(QLabel("Noise spectrum"))
		chart_heading.addStretch()
		self.peak_hold_checkbox = QCheckBox("Peak hold")
		self.peak_hold_checkbox.setChecked(True)
		self.peak_hold_checkbox.setToolTip("Show the highest level at each frequency since measurement start or the last peak reset.")
		chart_heading.addWidget(self.peak_hold_checkbox)
		right_layout.addLayout(chart_heading)
		self.spectra = SpectrumPanel()
		self.peak_hold_checkbox.toggled.connect(self.spectra.set_peak_hold_visible)
		right_layout.addWidget(self.spectra)
		self.splitter.addWidget(left)
		self.splitter.addWidget(right)
		self.splitter.setStretchFactor(0, 0)
		self.splitter.setStretchFactor(1, 1)
		self.splitter.setSizes([350, 1000])
		layout.addWidget(self.splitter, 1)
		self.state_label = QLabel("Idle")
		self.fps_label = QLabel("FPS 0.0")
		self.fps_label.setMinimumWidth(self.fps_label.fontMetrics().horizontalAdvance("FPS 999.9") + 4)
		self.fps_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
		self.fps_label.setToolTip("New audio display updates per second, averaged over one second.")
		self.elapsed_label = QLabel("Measured 00:00:00")
		self.statusBar().addWidget(self.state_label)
		self.statusBar().addPermanentWidget(self.fps_label)
		self.statusBar().addPermanentWidget(self.elapsed_label)
		self._set_running(False)

	def _set_running(self, running: bool) -> None:
		self.icon_animator.set_running(running)
		self.start_button.setText("Stop" if running else "Start")
		icon = QStyle.StandardPixmap.SP_MediaStop if running else QStyle.StandardPixmap.SP_MediaPlay
		self.start_button.setIcon(self.style().standardIcon(icon))

	def _reset_fps(self) -> None:
		self._fps_frames = 0
		self._fps_started = perf_counter()
		self.fps_label.setText("FPS 0.0")

	def _update_fps(self, rendered: bool) -> None:
		self._fps_frames += int(rendered)
		now = perf_counter()
		elapsed = now - self._fps_started
		if elapsed >= 1.0:
			self.fps_label.setText(f"FPS {self._fps_frames / elapsed:.1f}")
			self._fps_frames = 0
			self._fps_started = now

	def _show_notice(self, message: str = "") -> None:
		self.notice.setText(message)
		self.notice.setVisible(bool(message))

	def _set_channel_enabled(self, channel: int, enabled: bool) -> None:
		device = self.device_combo.currentData()
		if not isinstance(device, InputDevice):
			return
		disabled_channels = self._disabled_channels.setdefault((device.host_api, device.name), set())
		if enabled:
			disabled_channels.discard(channel)
		else:
			disabled_channels.add(channel)
		self.spectra.set_channel_enabled(channel, enabled)

	def _show_channels(self, count: int, sample_rate: float) -> None:
		device = self.device_combo.currentData()
		disabled_channels = self._disabled_channels.get((device.host_api, device.name), set()) if isinstance(device, InputDevice) else set()
		while self.channel_layout.count():
			item = self.channel_layout.takeAt(0)
			widget = item.widget()
			if widget is not None:
				widget.deleteLater()
		self.channel_widgets.clear()
		for channel in range(count):
			widget = ChannelWidget(channel)
			widget.enabled_checkbox.setChecked(channel not in disabled_channels)
			widget.enabled_changed.connect(self._set_channel_enabled)
			widget.sensor_commands_requested.connect(self._configure_commands)
			self.channel_layout.addWidget(widget)
			self.channel_widgets.append(widget)
		self.commands_button.setEnabled(count > 0)
		if self.channel_widgets:
			minimum_width = self.channel_widgets[0].minimumSizeHint().width() + 24
			minimum_width += self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
			self.splitter.widget(0).setMinimumWidth(max(292, minimum_width))
		if not count:
			self.channel_layout.addWidget(QLabel("No input channels"))
		self.channel_layout.addStretch()
		self.channel_label.setText(str(count))
		self.spectra.set_channels(count, sample_rate)
		for channel in disabled_channels:
			self.spectra.set_channel_enabled(channel, False)
		self.elapsed_label.setText("Measured 00:00:00")
		self._last_frames = -1
		self._last_warning = ""

	def refresh_devices(self) -> None:
		previous = self.device_combo.currentData()
		self._stop_capture()
		error = ""
		try:
			devices = sorted(
				query_input_devices(),
				key=lambda device: (device.name.casefold(), device.host_api.casefold(), device.index),
			)
			default_index = int(sd.default.device[0])
		except (sd.PortAudioError, OSError) as exception:
			devices = []
			default_index = -1
			error = f"Audio device discovery failed: {exception}"
		selected_index = previous.index if isinstance(previous, InputDevice) else default_index
		selected_row = 0
		with QSignalBlocker(self.device_combo):
			self.device_combo.clear()
			for row, device in enumerate(devices):
				self.device_combo.addItem(f"{row + 1}. {device.label}", device)
				if device.index == selected_index:
					selected_row = row
			if devices:
				self.device_combo.setCurrentIndex(selected_row)
		for widget in (
			self.device_combo, self.sample_rate, self.channel_count, self.fft_size,
			self.latency, self.apply_button, self.start_button, self.reset_button,
		):
			widget.setEnabled(bool(devices))
		if devices:
			self._select_device()
		else:
			self.analyzer = None
			self._show_channels(0, 48000)
			self.state_label.setText("No input device")
			self._show_notice(error or "No audio input device is available. Connect a microphone and refresh devices.")

	def _select_device(self, _index: int = 0) -> None:
		device = self.device_combo.currentData()
		if not isinstance(device, InputDevice):
			return
		self._stop_capture()
		channels = min(device.channels, DEFAULT_MAX_CHANNELS)
		self.sample_rate.setValue(device.sample_rate)
		self.channel_count.setMaximum(device.channels)
		self.channel_count.setValue(channels)
		block_size = min(FFT_SIZES[-1], max(FFT_SIZES[0], 2 ** round(math.log2(device.sample_rate * 0.05))))
		self.fft_size.setCurrentIndex(self.fft_size.findData(block_size))
		self.latency.setValue(max(0.1, device.latency * 1000))
		self._show_channels(channels, device.sample_rate)

	def start_capture(self) -> None:
		device = self.device_combo.currentData()
		if not isinstance(device, InputDevice):
			return
		self._stop_capture()
		settings = AudioSettings(
			self.sample_rate.value(),
			self.channel_count.value(),
			int(self.fft_size.currentData()),
			self.latency.value() / 1000,
		)
		self._show_channels(settings.channels, settings.sample_rate)
		self._show_notice()
		self.analyzer = SpectrumAnalyzer(settings.sample_rate, settings.block_size)
		try:
			self.capture.start(device, settings)
		except (sd.PortAudioError, ValueError, OSError) as exception:
			self.state_label.setText("Unavailable")
			self._show_notice(f"Cannot start audio input: {exception}. Check microphone permissions and capture settings.")
			return
		self._reset_fps()
		self._set_running(True)
		self.state_label.setText(
			f"Live | {settings.sample_rate:,.0f} Hz | {settings.channels} ch | {settings.block_size:,} samples"
		)

	def _stop_capture(self) -> None:
		self.capture.stop()
		self._reset_fps()
		self._set_running(False)
		self.state_label.setText("Stopped")

	def _toggle_capture(self) -> None:
		if self.capture.stream is not None:
			self._stop_capture()
		else:
			self.start_capture()

	def _update_measurement(self) -> None:
		stream = self.capture.stream
		if stream is None:
			return
		try:
			active = stream.active
		except sd.PortAudioError:
			active = False
		if not active:
			self._stop_capture()
			self._show_notice("Audio input stopped unexpectedly. Check the device and restart capture.")
			return
		previous_frames = self._last_frames
		self._render_measurement()
		self._update_fps(self._last_frames != previous_frames)

	def _render_measurement(self) -> None:
		buffer = self.capture.buffer
		settings = self.capture.settings
		if buffer is None or settings is None or self.analyzer is None:
			return
		frame = buffer.snapshot()
		if frame is None or frame.frames == self._last_frames:
			return
		self._last_frames = frame.frames
		for channel, widget in enumerate(self.channel_widgets):
			widget.update_levels(float(frame.levels[channel]), float(frame.peaks[channel]))
		if frame.spectrum is not None and frame.spectrum_peaks is not None:
			self.spectra.update_spectra(self.analyzer.frequencies, frame.spectrum, frame.spectrum_peaks)
		seconds = int(frame.frames / settings.sample_rate)
		hours, remainder = divmod(seconds, 3600)
		minutes, seconds = divmod(remainder, 60)
		self.elapsed_label.setText(f"Measured {hours:02d}:{minutes:02d}:{seconds:02d}")
		if frame.warning and frame.warning != self._last_warning:
			self._last_warning = frame.warning
			self._show_notice(f"Audio warning: {frame.warning.strip()}")

	def _reset_measurement(self) -> None:
		if self.capture.buffer is not None:
			self.capture.buffer.reset_peaks()
			self._last_frames = -1
			self._last_warning = ""
			self._show_notice()
			self._render_measurement()

	def _command_fields(self, device: InputDevice, channel: int) -> tuple[str, QLineEdit, QLineEdit]:
		device_key = f"{quote(device.host_api, safe='')}/{quote(device.name, safe='')}"
		prefix = f"commands/devices/{device_key}/channels/{channel + 1}"
		legacy_temperature = self.settings.value("commands/temperature", "") if channel == 0 else ""
		legacy_speed = self.settings.value("commands/speed", "") if channel == 0 else ""
		temperature = QLineEdit(str(self.settings.value(f"{prefix}/temperature", legacy_temperature)))
		speed = QLineEdit(str(self.settings.value(f"{prefix}/speed", legacy_speed)))
		temperature.setObjectName("temperatureCommand")
		speed.setObjectName("speedCommand")
		for field in (temperature, speed):
			field.setPlaceholderText("Not configured")
			field.setToolTip("Saved locally only. Command execution is reserved for a future version.")
		return prefix, temperature, speed

	def _save_command_fields(self, command_fields: list[tuple[str, QLineEdit, QLineEdit]]) -> None:
		for prefix, temperature, speed in command_fields:
			self.settings.setValue(f"{prefix}/temperature", temperature.text())
			self.settings.setValue(f"{prefix}/speed", speed.text())
		self.settings.remove("commands/temperature")
		self.settings.remove("commands/speed")

	def _configure_all_commands(self) -> None:
		device = self.device_combo.currentData()
		if not isinstance(device, InputDevice) or not self.channel_widgets:
			return
		dialog = QDialog(self)
		dialog.setWindowTitle("Sensor commands - All channels")
		dialog.resize(980, min(640, 180 + 40 * len(self.channel_widgets)))
		layout = QVBoxLayout(dialog)
		header = QFormLayout()
		device_label = QLabel(device.label)
		device_label.setTextFormat(Qt.TextFormat.PlainText)
		device_label.setWordWrap(True)
		header.addRow("Device", device_label)
		header.addRow("Status", QLabel("Inactive"))
		layout.addLayout(header)
		table = QTableWidget(len(self.channel_widgets), 3)
		table.setObjectName("allCommandChannels")
		table.setHorizontalHeaderLabels(["Channel", "Temperature command", "Fan speed command"])
		table.verticalHeader().hide()
		table.verticalHeader().setDefaultSectionSize(40)
		table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
		for column in (1, 2):
			table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
		table.setAlternatingRowColors(True)
		table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
		table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
		command_fields = []
		for channel in range(len(self.channel_widgets)):
			fields = self._command_fields(device, channel)
			_, temperature, speed = fields
			table.setItem(channel, 0, QTableWidgetItem(f"Channel {channel + 1:02d}"))
			table.setCellWidget(channel, 1, temperature)
			table.setCellWidget(channel, 2, speed)
			command_fields.append(fields)
		layout.addWidget(table)
		buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
		buttons.accepted.connect(dialog.accept)
		buttons.rejected.connect(dialog.reject)
		layout.addWidget(buttons)
		if dialog.exec() == QDialog.DialogCode.Accepted:
			self._save_command_fields(command_fields)

	def _configure_commands(self, channel: int = 0) -> None:
		device = self.device_combo.currentData()
		if not isinstance(device, InputDevice) or not 0 <= channel < len(self.channel_widgets):
			return
		dialog = QDialog(self)
		dialog.setWindowTitle("Sensor commands")
		dialog.resize(640, 260)
		layout = QVBoxLayout(dialog)
		header = QFormLayout()
		device_label = QLabel(device.label)
		device_label.setTextFormat(Qt.TextFormat.PlainText)
		device_label.setWordWrap(True)
		header.addRow("Device", device_label)
		channel_combo = QComboBox()
		channel_combo.setObjectName("commandChannel")
		header.addRow("Channel", channel_combo)
		layout.addLayout(header)
		pages = QStackedWidget()
		pages.setObjectName("commandPages")
		command_fields: list[tuple[str, QLineEdit, QLineEdit]] = []
		for index in range(len(self.channel_widgets)):
			channel_combo.addItem(f"Channel {index + 1:02d}", index)
			prefix, temperature, speed = self._command_fields(device, index)
			page = QWidget()
			form = QFormLayout(page)
			form.addRow("Temperature command", temperature)
			form.addRow("Fan speed command", speed)
			form.addRow("Status", QLabel("Inactive"))
			pages.addWidget(page)
			command_fields.append((prefix, temperature, speed))
		channel_combo.currentIndexChanged.connect(pages.setCurrentIndex)
		channel_combo.setCurrentIndex(channel)
		layout.addWidget(pages)
		buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
		buttons.accepted.connect(dialog.accept)
		buttons.rejected.connect(dialog.reject)
		layout.addWidget(buttons)
		if dialog.exec() == QDialog.DialogCode.Accepted:
			self._save_command_fields(command_fields)

	def closeEvent(self, event: QCloseEvent) -> None:
		self.timer.stop()
		self.icon_animator.set_running(False)
		self.capture.stop()
		self.settings.setValue("window/geometry", self.saveGeometry())
		self.settings.setValue("window/splitter", self.splitter.saveState())
		super().closeEvent(event)


def configure_application(application: QApplication) -> None:
	application.setApplicationName("JustFan")
	application.setOrganizationName("JustFan")
	application.setWindowIcon(QIcon(str(Path(__file__).with_name("JustFan.ico"))))
	application.setStyle("Fusion")
	application.setFont(QFont("Segoe UI" if sys.platform == "win32" else "Noto Sans", 10))
	palette = QPalette()
	for role, color in (
		(QPalette.ColorRole.Window, "#f1f4f6"),
		(QPalette.ColorRole.WindowText, "#26343b"),
		(QPalette.ColorRole.Base, "#ffffff"),
		(QPalette.ColorRole.AlternateBase, "#edf2f3"),
		(QPalette.ColorRole.Text, "#26343b"),
		(QPalette.ColorRole.Button, "#ffffff"),
		(QPalette.ColorRole.ButtonText, "#26343b"),
		(QPalette.ColorRole.Highlight, "#087f8c"),
		(QPalette.ColorRole.HighlightedText, "#ffffff"),
		(QPalette.ColorRole.ToolTipBase, "#ffffff"),
		(QPalette.ColorRole.ToolTipText, "#26343b"),
	):
		palette.setColor(role, QColor(color))
	application.setPalette(palette)
	application.setStyleSheet("""
		QWidget#controls { background: #ffffff; border-bottom: 1px solid #d9e1e5; }
		QFrame#channel { background: #ffffff; border: 1px solid #d9e1e5; border-radius: 6px; }
		QLabel#notice { background: #fff0e6; color: #893816; }
		QSplitter::handle { background: #d9e1e5; }
		QSplitter::handle:hover { background: #087f8c; }
		QPushButton, QToolButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
			min-height: 26px;
		}
		QPushButton { padding: 2px 10px; }
		QToolButton { padding: 2px 6px; }
		QStatusBar { background: #ffffff; border-top: 1px solid #d9e1e5; }
	""")
	pg.setConfigOptions(antialias=False, foreground="#46545d")


def main() -> int:
	application = QApplication(sys.argv)
	configure_application(application)
	window = MainWindow()
	window.show()
	interrupt_timer = QTimer()
	interrupt_timer.setInterval(500)
	interrupt_timer.timeout.connect(lambda: None)
	interrupt_timer.start()
	previous_handler = signal.signal(signal.SIGINT, lambda _signum, _frame: window.close())
	try:
		return application.exec()
	finally:
		interrupt_timer.stop()
		signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
	raise SystemExit(main())
