"""Deterministic signal and offscreen UI tests; no microphone is opened."""

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QDialogButtonBox, QLabel, QLineEdit, QStackedWidget, QTableWidget, QWidget

import JustFan as justfan


class FakeStream:
    def __init__(self, **options) -> None:
        self.callback = options["callback"]
        self.active = False
        self.closed = False

    def start(self) -> None:
        self.active = True

    def stop(self) -> None:
        self.active = False

    def close(self) -> None:
        self.active = False
        self.closed = True


class SignalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_rate = 48000
        self.size = 4096
        time = np.arange(self.size) / self.sample_rate
        self.signal = np.column_stack((0.5 * np.sin(2 * np.pi * 100 * time * self.sample_rate / self.size), np.zeros(self.size)))
        self.expected = 20 * np.log10(0.5 / np.sqrt(2))

    def test_rms_and_silent_channel(self) -> None:
        buffer = justfan.CaptureBuffer(2)
        buffer.push(self.signal)
        frame = buffer.snapshot()
        self.assertIsNotNone(frame)
        assert frame is not None
        np.testing.assert_allclose(frame.levels, [self.expected, justfan.DB_FLOOR], atol=1e-6)
        self.assertEqual(frame.frames, self.size)

    def test_peak_survives_unrendered_blocks_and_reset(self) -> None:
        buffer = justfan.CaptureBuffer(2)
        self.assertIsNone(buffer.snapshot())
        buffer.push(self.signal, "input overflow")
        buffer.push(self.signal * 0.1)
        frame = buffer.snapshot()
        assert frame is not None
        self.assertAlmostEqual(float(frame.levels[0]), self.expected - 20)
        self.assertAlmostEqual(float(frame.peaks[0]), self.expected)
        self.assertEqual(frame.warning, "input overflow")
        buffer.reset_peaks()
        reset = buffer.snapshot()
        assert reset is not None
        np.testing.assert_array_equal(reset.peaks, frame.levels)
        self.assertEqual(reset.frames, 0)
        self.assertEqual(reset.warning, "")

    def test_capture_owns_its_samples(self) -> None:
        buffer = justfan.CaptureBuffer(2)
        original = self.signal.copy()
        buffer.push(original)
        original.fill(0)
        frame = buffer.snapshot()
        assert frame is not None
        np.testing.assert_array_equal(frame.samples, self.signal)

    def test_spectrum_frequency_and_rms_amplitude(self) -> None:
        analyzer = justfan.SpectrumAnalyzer(self.sample_rate, self.size)
        levels = analyzer.analyze(self.signal)
        self.assertEqual(levels.shape, (self.size // 2 + 1, 2))
        self.assertEqual(int(np.argmax(levels[:, 0])), 100)
        self.assertAlmostEqual(float(levels[100, 0]), self.expected, places=4)
        self.assertEqual(analyzer.frequencies[100], 100 * self.sample_rate / self.size)
        self.assertTrue(np.isfinite(levels).all())

    def test_spectrum_dc_and_nyquist_scaling(self) -> None:
        analyzer = justfan.SpectrumAnalyzer(self.sample_rate, self.size)
        dc = np.full((self.size, 1), 0.25)
        self.assertAlmostEqual(float(analyzer.analyze(dc)[0, 0]), 20 * np.log10(0.25))
        nyquist = (0.25 * (-1.0) ** np.arange(self.size))[:, None]
        self.assertAlmostEqual(float(analyzer.analyze(nyquist)[-1, 0]), 20 * np.log10(0.25))

    def test_spectrum_peak_hold_retains_unrendered_bins_per_channel(self) -> None:
        analyzer = justfan.SpectrumAnalyzer(self.sample_rate, self.size)
        buffer = justfan.CaptureBuffer(2, analyzer)
        phase = 2 * np.pi * np.arange(self.size) / self.size
        later = np.column_stack((0.05 * np.sin(200 * phase), 0.1 * np.sin(300 * phase)))
        first_spectrum = analyzer.analyze(self.signal)
        later_spectrum = analyzer.analyze(later)
        buffer.push(self.signal)
        buffer.push(later)
        frame = buffer.snapshot()
        assert frame is not None and frame.spectrum is not None and frame.spectrum_peaks is not None
        np.testing.assert_allclose(frame.spectrum, later_spectrum)
        np.testing.assert_allclose(frame.spectrum_peaks, np.maximum(first_spectrum, later_spectrum))
        self.assertGreater(float(frame.spectrum_peaks[100, 0]), float(frame.spectrum[100, 0]) + 40)
        self.assertAlmostEqual(float(frame.spectrum_peaks[300, 1]), 20 * np.log10(0.1 / np.sqrt(2)), places=4)

    def test_reset_spectrum_peak_hold_uses_current_spectrum(self) -> None:
        analyzer = justfan.SpectrumAnalyzer(self.sample_rate, self.size)
        buffer = justfan.CaptureBuffer(2, analyzer)
        buffer.reset_peaks()
        buffer.push(self.signal)
        buffer.push(self.signal * 0.1)
        buffer.reset_peaks()
        frame = buffer.snapshot()
        assert frame is not None and frame.spectrum is not None and frame.spectrum_peaks is not None
        np.testing.assert_array_equal(frame.spectrum_peaks, frame.spectrum)
        self.assertEqual(frame.frames, 0)
        frame.spectrum_peaks.fill(0)
        fresh = buffer.snapshot()
        assert fresh is not None and fresh.spectrum_peaks is not None
        np.testing.assert_array_equal(fresh.spectrum_peaks, analyzer.analyze(self.signal * 0.1))

    def test_eight_channel_level_and_spectrum_peaks_are_independent(self) -> None:
        analyzer = justfan.SpectrumAnalyzer(self.sample_rate, self.size)
        buffer = justfan.CaptureBuffer(8, analyzer)
        bins = 40 + 37 * np.arange(8)
        amplitudes = 0.4 * 0.7 ** np.arange(8)
        phase = 2 * np.pi * np.arange(self.size)[:, None] / self.size
        strong = np.sin(phase * bins[None, :]) * amplitudes[None, :]
        buffer.push(strong)
        buffer.push(strong * 0.02)
        frame = buffer.snapshot()
        assert frame is not None and frame.spectrum is not None and frame.spectrum_peaks is not None
        expected = 20 * np.log10(amplitudes / np.sqrt(2))
        np.testing.assert_allclose(frame.peaks, expected, atol=1e-6)
        np.testing.assert_allclose(frame.levels, expected + 20 * np.log10(0.02), atol=1e-6)
        np.testing.assert_allclose(frame.spectrum_peaks[bins, np.arange(8)], expected, atol=1e-4)
        np.testing.assert_array_equal(np.argmax(frame.spectrum_peaks, axis=0), bins)
        self.assertTrue(np.all(frame.spectrum_peaks >= frame.spectrum))

    def test_invalid_spectrum_input(self) -> None:
        with self.assertRaises(ValueError):
            justfan.SpectrumAnalyzer(0, self.size)
        with self.assertRaises(ValueError):
            justfan.SpectrumAnalyzer(self.sample_rate, 2)
        with self.assertRaises(ValueError):
            justfan.SpectrumAnalyzer(self.sample_rate, self.size).analyze(np.zeros((128, 2)))


class CaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = justfan.InputDevice(3, "Test microphone", "Test API", 2, 48000, 0.01)
        self.settings = justfan.AudioSettings(48000, 2, 2048, 0.01)

    def test_start_callback_and_close(self) -> None:
        capture = justfan.AudioCapture()
        with patch.object(justfan.sd, "check_input_settings") as check, patch.object(justfan.sd, "InputStream", side_effect=FakeStream):
            capture.start(self.device, self.settings)
            stream = capture.stream
            assert stream is not None
            self.assertTrue(stream.active)
            check.assert_called_once_with(device=3, channels=2, samplerate=48000, dtype="float32")
            stream.callback(np.full((2048, 2), 0.5), 2048, None, "")
            assert capture.buffer is not None
            frame = capture.buffer.snapshot()
            assert frame is not None
            np.testing.assert_allclose(frame.levels, 20 * np.log10(0.5))
            self.assertIsNotNone(frame.spectrum)
            self.assertIsNotNone(frame.spectrum_peaks)
            capture.stop()
            self.assertTrue(stream.closed)
            self.assertIsNone(capture.stream)

    def test_failed_start_closes_stream_and_drops_previous_buffer(self) -> None:
        capture = justfan.AudioCapture()
        failed = FakeStream(callback=lambda *_args: None)
        with patch.object(justfan.sd, "check_input_settings"), patch.object(justfan.sd, "InputStream", side_effect=FakeStream):
            capture.start(self.device, self.settings)
        previous = capture.stream
        assert previous is not None
        with patch.object(justfan.sd, "check_input_settings"), patch.object(justfan.sd, "InputStream", return_value=failed), patch.object(failed, "start", side_effect=justfan.sd.PortAudioError("Device busy")):
            with self.assertRaises(justfan.sd.PortAudioError):
                capture.start(self.device, self.settings)
        self.assertTrue(previous.closed)
        self.assertTrue(failed.closed)
        self.assertIsNone(capture.stream)
        self.assertIsNone(capture.buffer)

    def test_stop_closes_even_if_driver_stop_fails(self) -> None:
        capture = justfan.AudioCapture()
        stream = FakeStream(callback=lambda *_args: None)
        capture.stream = stream
        with patch.object(stream, "stop", side_effect=justfan.sd.PortAudioError("Disconnected")):
            capture.stop()
        self.assertTrue(stream.closed)
        self.assertIsNone(capture.stream)


class InterfaceTests(unittest.TestCase):
    application = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])
        justfan.configure_application(cls.application)

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = QSettings(os.path.join(self.directory.name, "settings.ini"), QSettings.Format.IniFormat)
        self.devices = [
            justfan.InputDevice(3, "Stereo microphone", "Test API", 2, 48000, 0.01),
            justfan.InputDevice(8, "Mono microphone", "Test API", 1, 44100, 0.02),
        ]
        self.query = self.enterContext(patch.object(justfan, "query_input_devices", return_value=self.devices))
        self.enterContext(patch.object(justfan.sd, "default", SimpleNamespace(device=(3, -1))))
        self.check = self.enterContext(patch.object(justfan.sd, "check_input_settings"))
        self.enterContext(patch.object(justfan.sd, "InputStream", side_effect=FakeStream))
        self.window = justfan.MainWindow(self.settings)
        self.addCleanup(self.window.close)
        self.window.show()
        assert self.application is not None
        self.application.processEvents()

    def push(self, amplitude: float = 0.25) -> None:
        settings = self.window.capture.settings
        stream = self.window.capture.stream
        assert settings is not None and stream is not None
        stream.callback(np.full((settings.block_size, settings.channels), amplitude), settings.block_size, None, "")
        self.window._update_measurement()

    def test_fps_counts_only_new_display_frames_and_handles_no_new_audio(self) -> None:
        self.assertLess(self.window.fps_label.x(), self.window.elapsed_label.x())
        self.assertLessEqual(self.window.fps_label.geometry().right(), self.window.elapsed_label.x())
        with patch.object(justfan, "perf_counter", return_value=100.0) as clock:
            self.window.start_capture()
            clock.return_value = 100.25
            self.push()
            self.window._update_measurement()
            clock.return_value = 100.5
            self.push()
            clock.return_value = 101.0
            self.window._update_measurement()
            self.assertEqual(self.window.fps_label.text(), "FPS 2.0")
            clock.return_value = 102.0
            self.window._update_measurement()
            self.assertEqual(self.window.fps_label.text(), "FPS 0.0")

    def test_fps_resets_on_stop_restart_and_missing_device(self) -> None:
        with patch.object(justfan, "perf_counter", return_value=100.0) as clock:
            self.window.start_capture()
            clock.return_value = 101.0
            self.push()
            self.assertEqual(self.window.fps_label.text(), "FPS 1.0")
            self.window._toggle_capture()
            self.assertEqual(self.window.fps_label.text(), "FPS 0.0")
            clock.return_value = 102.0
            self.window._toggle_capture()
            self.assertEqual(self.window.fps_label.text(), "FPS 0.0")
            clock.return_value = 103.0
            self.push()
            self.assertEqual(self.window.fps_label.text(), "FPS 1.0")
            self.query.return_value = []
            self.window.refresh_devices()
            self.assertEqual(self.window.fps_label.text(), "FPS 0.0")

    def test_compact_controls_remove_duplicate_title_and_reuse_parameter_row(self) -> None:
        controls = self.window.findChild(QWidget, "controls")
        assert controls is not None
        control_layout = controls.layout()
        assert control_layout is not None
        self.assertEqual(self.window.windowTitle(), "JustFan v1.0")
        self.assertFalse(any(label.text() == "JustFan" for label in controls.findChildren(QLabel)))
        self.assertEqual(control_layout.count(), 2)
        parameters = control_layout.itemAt(1).layout()
        assert parameters is not None
        self.assertIs(parameters.itemAtPosition(0, 4).widget(), self.window.commands_button)
        self.assertIs(parameters.itemAtPosition(1, 4).widget(), self.window.apply_button)
        assert self.application is not None
        for width in (1360, 860):
            with self.subTest(width=width):
                self.window.resize(width, 680)
                self.application.processEvents()
                self.assertGreaterEqual(self.window.width(), controls.minimumSizeHint().width())
                widgets = (
                    self.window.sample_rate, self.window.channel_count, self.window.fft_size,
                    self.window.latency, self.window.commands_button, self.window.apply_button,
                )
                for index, widget in enumerate(widgets):
                    self.assertTrue(controls.rect().contains(widget.geometry()))
                    for other in widgets[index + 1:]:
                        self.assertFalse(widget.geometry().intersects(other.geometry()))
                self.assertGreater(self.window.commands_button.y(), self.window.device_combo.geometry().bottom())
                self.assertLess(self.window.commands_button.geometry().bottom(), self.window.apply_button.y())
                self.assertGreaterEqual(self.window.commands_button.width(), self.window.commands_button.sizeHint().width())

    def test_current_peak_placeholders_and_graphs(self) -> None:
        self.push(0.5)
        self.push(0.25)
        for widget in self.window.channel_widgets:
            self.assertEqual(widget.current_values[0].text(), "-12.0")
            self.assertEqual(widget.peak_values[0].text(), "-6.0")
            self.assertTrue(widget.current_values[0].alignment() & Qt.AlignmentFlag.AlignLeft)
            self.assertTrue(widget.peak_values[0].alignment() & Qt.AlignmentFlag.AlignRight)
            for label in widget.current_values[1:] + widget.peak_values[1:]:
                self.assertEqual(label.text(), "--")
        settings = self.window.capture.settings
        assert settings is not None
        self.assertEqual(len(self.window.spectra.curves[0].getData()[0]), settings.block_size // 2 + 1)

    def test_spectrum_peak_hold_plots_unrendered_frequency_peaks(self) -> None:
        settings = self.window.capture.settings
        stream = self.window.capture.stream
        analyzer = self.window.analyzer
        assert settings is not None and stream is not None and analyzer is not None
        phase = 2 * np.pi * np.arange(settings.block_size) / settings.block_size
        first = np.column_stack((0.5 * np.sin(32 * phase), 0.25 * np.sin(64 * phase)))
        later = np.column_stack((0.05 * np.sin(96 * phase), 0.1 * np.sin(128 * phase)))
        stream.callback(first, settings.block_size, None, "")
        stream.callback(later, settings.block_size, None, "")
        self.window._update_measurement()
        current = analyzer.analyze(later)
        held = np.maximum(analyzer.analyze(first), current)
        self.assertTrue(self.window.peak_hold_checkbox.isChecked())
        for channel in range(settings.channels):
            np.testing.assert_allclose(self.window.spectra.curves[channel].getData()[1], current[:, channel])
            np.testing.assert_allclose(self.window.spectra.peak_curves[channel].getData()[1], held[:, channel])

    def test_spectrum_peak_visibility_keeps_history_and_survives_restart(self) -> None:
        self.push(0.5)
        self.window.peak_hold_checkbox.setChecked(False)
        self.push(0.25)
        for curve in self.window.spectra.peak_curves:
            self.assertFalse(curve.isVisible())
            self.assertAlmostEqual(float(curve.getData()[1][0]), 20 * np.log10(0.5))
        self.window.peak_hold_checkbox.setChecked(True)
        for curve in self.window.spectra.peak_curves:
            self.assertTrue(curve.isVisible())
            self.assertAlmostEqual(float(curve.getData()[1][0]), 20 * np.log10(0.5))
        self.window.peak_hold_checkbox.setChecked(False)
        self.window.start_capture()
        for curve in self.window.spectra.peak_curves:
            self.assertFalse(curve.isVisible())
            self.assertIsNone(curve.getData()[0])
        self.window.peak_hold_checkbox.setChecked(True)
        self.push(0.1)
        for curve in self.window.spectra.peak_curves:
            self.assertTrue(curve.isVisible())
            self.assertAlmostEqual(float(curve.getData()[1][0]), 20 * np.log10(0.1))

    def test_reset_peaks_resets_held_spectrum_while_stopped(self) -> None:
        self.push(0.5)
        self.push(0.25)
        self.window._toggle_capture()
        self.window.reset_button.click()
        for current, held in zip(self.window.spectra.curves, self.window.spectra.peak_curves):
            np.testing.assert_array_equal(held.getData()[1], current.getData()[1])
            self.assertAlmostEqual(float(held.getData()[1][0]), 20 * np.log10(0.25))
        self.assertEqual(self.window.elapsed_label.text(), "Measured 00:00:00")

    def test_eight_channel_floor_peaks_remain_visible(self) -> None:
        self.query.return_value = [justfan.InputDevice(3, "Eight-channel microphone", "Test API", 8, 48000, 0.01)]
        self.window.refresh_devices()
        settings = self.window.capture.settings
        stream = self.window.capture.stream
        assert settings is not None and stream is not None
        signal = np.zeros((settings.block_size, 8))
        phase = 2 * np.pi * np.arange(settings.block_size) / settings.block_size
        signal[:, 0] = 0.5 * np.sin(32 * phase)
        signal[:, 1] = 0.25 * np.sin(64 * phase)
        stream.callback(signal, settings.block_size, None, "")
        stream.callback(np.zeros_like(signal), settings.block_size, None, "")
        self.window._update_measurement()
        assert self.window.capture.buffer is not None
        frame = self.window.capture.buffer.snapshot()
        assert frame is not None and frame.spectrum_peaks is not None
        self.assertEqual(len(self.window.spectra.peak_curves), 8)
        for channel, held in enumerate(self.window.spectra.peak_curves):
            with self.subTest(channel=channel + 1, check="data"):
                self.assertTrue(held.isVisible())
                np.testing.assert_array_equal(held.getData()[1], frame.spectrum_peaks[:, channel])
            with self.subTest(channel=channel + 1, check="overlap"):
                self.assertGreater(held.zValue(), self.window.spectra.curves[channel].zValue())
                self.assertGreater(held.opts["pen"].widthF(), self.window.spectra.curves[channel].opts["pen"].widthF())
            with self.subTest(channel=channel + 1, check="floor"):
                self.assertLess(self.window.spectra.plots[channel].getViewBox().viewRange()[1][0], justfan.DB_FLOOR)

    def test_device_sorting_preserves_default_and_refresh_selection(self) -> None:
        devices = [
            justfan.InputDevice(12, "USB microphone", "Windows WASAPI", 2, 48000, 0.01),
            justfan.InputDevice(9, "alpha microphone", "Windows WASAPI", 1, 44100, 0.02),
            self.devices[0],
            justfan.InputDevice(6, "Alpha microphone", "ALSA", 1, 44100, 0.02),
            justfan.InputDevice(11, "USB microphone", "MME", 2, 48000, 0.01),
        ]
        self.query.return_value = devices
        self.window.refresh_devices()
        combo = self.window.device_combo
        self.assertEqual([combo.itemData(row).index for row in range(combo.count())], [6, 9, 3, 11, 12])
        self.assertEqual(
            [combo.itemText(row) for row in range(combo.count())],
            [f"{row + 1}. {combo.itemData(row).label}" for row in range(combo.count())],
        )
        self.assertEqual(combo.currentData().index, 3)
        self.assertEqual(combo.currentText(), f"3. {self.devices[0].label}")
        self.assertEqual(self.check.call_args.kwargs["device"], 3)
        combo.setCurrentIndex(combo.findText(devices[0].label, Qt.MatchFlag.MatchEndsWith))
        self.window.refresh_devices()
        self.assertEqual(combo.currentData().index, 12)
        self.assertEqual(combo.currentText(), f"5. {devices[0].label}")
        self.assertEqual(self.check.call_args.kwargs["device"], 12)

    def test_device_switch_clears_readings_and_queries_parameters(self) -> None:
        self.push()
        previous = self.window.capture.stream
        assert previous is not None
        self.window.device_combo.setCurrentIndex(self.window.device_combo.findText(self.devices[1].label, Qt.MatchFlag.MatchEndsWith))
        self.assertTrue(previous.closed)
        self.assertEqual(len(self.window.channel_widgets), 1)
        self.assertEqual(len(self.window.spectra.curves), 1)
        self.assertEqual(len(self.window.spectra.peak_curves), 1)
        self.assertEqual(self.window.channel_widgets[0].current_values[0].text(), "--")
        self.assertEqual(self.window.sample_rate.value(), 44100)
        self.assertEqual(self.window.channel_count.value(), 1)
        self.assertEqual(self.window.latency.value(), 20)
        self.assertIsNone(self.window.spectra.curves[0].getData()[0])
        self.assertIsNone(self.window.spectra.peak_curves[0].getData()[0])

    def test_double_click_fills_spectrum_panel_and_restores_grid(self) -> None:
        assert self.application is not None
        panel = self.window.spectra
        plot = panel.plots[1]
        curves = list(panel.curves)
        self.push(0.5)
        QTest.mouseDClick(plot.viewport(), Qt.MouseButton.LeftButton, pos=plot.viewport().rect().center())
        self.application.processEvents()
        self.assertEqual([item.isVisible() for item in panel.plots], [False, True])
        self.assertEqual(panel.grid.count(), 1)
        self.assertEqual(plot.geometry(), panel.viewport().rect().marginsRemoved(panel.grid.contentsMargins()))
        self.assertEqual(panel.curves, curves)
        self.push(0.25)
        QTest.mouseDClick(plot.viewport(), Qt.MouseButton.LeftButton, pos=plot.viewport().rect().center())
        self.application.processEvents()
        self.assertTrue(all(item.isVisible() for item in panel.plots))
        self.assertEqual(panel.grid.count(), 2)
        for current, held in zip(panel.curves, panel.peak_curves):
            self.assertAlmostEqual(float(current.getData()[1][0]), 20 * np.log10(0.25))
            self.assertAlmostEqual(float(held.getData()[1][0]), 20 * np.log10(0.5))

    def test_focused_spectrum_fills_panel_after_scrolling_and_resizing(self) -> None:
        assert self.application is not None
        self.query.return_value = [justfan.InputDevice(3, "Eight-channel microphone", "Test API", 8, 48000, 0.01)]
        self.window.refresh_devices()
        self.application.processEvents()
        panel = self.window.spectra
        panel.verticalScrollBar().setValue(panel.verticalScrollBar().maximum())
        plot = panel.plots[-1]
        QTest.mouseDClick(plot.viewport(), Qt.MouseButton.LeftButton, pos=plot.viewport().rect().center())
        self.application.processEvents()
        for width, sizes in ((1360, [300, 1050]), (860, [500, 360]), (1500, [350, 1150])):
            with self.subTest(width=width):
                self.window.resize(width, 680)
                self.window.splitter.setSizes(sizes)
                self.application.processEvents()
                self.assertEqual(panel.columns, 1)
                self.assertEqual(panel.grid.count(), 1)
                self.assertEqual([item.isVisible() for item in panel.plots], [False] * 7 + [True])
                self.assertFalse(panel.verticalScrollBar().isVisible())
                self.assertFalse(panel.horizontalScrollBar().isVisible())
                self.assertEqual(plot.geometry(), panel.viewport().rect().marginsRemoved(panel.grid.contentsMargins()))
        QTest.mouseDClick(plot.viewport(), Qt.MouseButton.LeftButton, pos=plot.viewport().rect().center())
        self.application.processEvents()
        self.assertTrue(all(item.isVisible() for item in panel.plots))
        self.assertEqual(panel.grid.getItemPosition(panel.grid.indexOf(plot))[:2], (7 // panel.columns, 7 % panel.columns))

    def test_rebuilding_channels_clears_focused_spectrum(self) -> None:
        assert self.application is not None
        panel = self.window.spectra
        plot = panel.plots[1]
        QTest.mouseDClick(plot.viewport(), Qt.MouseButton.LeftButton, pos=plot.viewport().rect().center())
        self.window.device_combo.setCurrentIndex(self.window.device_combo.findText(self.devices[1].label, Qt.MatchFlag.MatchEndsWith))
        self.application.processEvents()
        self.assertIsNone(panel._focused_plot)
        self.assertEqual(len(panel.plots), 1)
        self.assertTrue(panel.plots[0].isVisible())
        plot = panel.plots[0]
        QTest.mouseDClick(plot.viewport(), Qt.MouseButton.LeftButton, pos=plot.viewport().rect().center())
        self.application.processEvents()
        self.assertEqual(plot.geometry(), panel.viewport().rect().marginsRemoved(panel.grid.contentsMargins()))
        self.query.return_value = []
        self.window.refresh_devices()
        self.application.processEvents()
        self.assertIsNone(panel._focused_plot)
        self.assertEqual(panel.plots, [])
        self.assertEqual(panel.grid.count(), 1)
        self.assertEqual(panel.grid.itemAt(0).widget().text(), "No spectrum data")
        self.query.return_value = self.devices
        self.window.refresh_devices()
        self.application.processEvents()
        self.assertEqual(len(panel.plots), 2)
        self.assertTrue(all(item.isVisible() for item in panel.plots))

    def test_splitter_reflows_graphs_and_channel_values(self) -> None:
        assert self.application is not None
        self.window.splitter.setSizes([800, 400])
        self.application.processEvents()
        self.assertEqual(self.window.spectra.columns, 1)
        self.push()
        self.window.splitter.setSizes([300, 1050])
        self.application.processEvents()
        self.assertEqual(self.window.spectra.columns, 2)
        for widget in self.window.channel_widgets:
            for label in widget.current_values + widget.peak_values:
                self.assertGreaterEqual(label.width(), label.fontMetrics().horizontalAdvance(label.text()))

    def test_edit_apply_stop_and_reset(self) -> None:
        self.window.sample_rate.setValue(32000)
        self.window.channel_count.setValue(1)
        self.window.fft_size.setCurrentIndex(self.window.fft_size.findData(4096))
        self.window.latency.setValue(30)
        self.window.start_capture()
        self.assertEqual(self.window.capture.settings, justfan.AudioSettings(32000, 1, 4096, 0.03))
        self.push(0.5)
        self.push(0.25)
        self.window._toggle_capture()
        self.assertIsNone(self.window.capture.stream)
        self.window._reset_measurement()
        self.assertEqual(self.window.channel_widgets[0].peak_values[0].text(), "-12.0")
        self.assertEqual(self.window.elapsed_label.text(), "Measured 00:00:00")

    def test_unsupported_settings_are_reported(self) -> None:
        self.check.side_effect = justfan.sd.PortAudioError("Unsupported sample rate")
        self.window.start_capture()
        self.assertIsNone(self.window.capture.stream)
        self.assertEqual(self.window.state_label.text(), "Unavailable")
        self.assertIn("Unsupported sample rate", self.window.notice.text())
        self.assertEqual(self.window.start_button.text(), "Start")

    def test_no_devices_clears_channels_and_can_recover(self) -> None:
        self.query.return_value = []
        self.window.refresh_devices()
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertFalse(self.window.commands_button.isEnabled())
        with patch.object(justfan, "QDialog") as dialog:
            self.window._configure_all_commands()
            dialog.assert_not_called()
        self.assertEqual(self.window.channel_widgets, [])
        self.assertEqual(self.window.spectra.curves, [])
        self.assertEqual(self.window.spectra.peak_curves, [])
        self.query.return_value = self.devices
        self.window.refresh_devices()
        self.assertTrue(self.window.start_button.isEnabled())
        self.assertTrue(self.window.commands_button.isEnabled())
        self.assertEqual(len(self.window.channel_widgets), 2)

    def test_disconnect_is_reported(self) -> None:
        stream = self.window.capture.stream
        assert stream is not None
        stream.active = False
        self.window._update_measurement()
        self.assertTrue(stream.closed)
        self.assertIn("stopped unexpectedly", self.window.notice.text())

    def edit_sensor_commands(self, interaction, opener=None) -> None:
        failures = []

        def edit() -> None:
            assert self.application is not None
            dialog = self.application.activeModalWidget()
            if not isinstance(dialog, QDialog):
                failures.append(AssertionError("The sensor command dialog did not open."))
                return
            try:
                interaction(dialog)
            except Exception as exception:
                failures.append(exception)
            finally:
                if dialog.isVisible():
                    dialog.reject()

        def timeout() -> None:
            failures.append(AssertionError("The sensor command dialog did not close."))
            assert self.application is not None
            dialog = self.application.activeModalWidget()
            if isinstance(dialog, QDialog):
                dialog.reject()

        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(timeout)
        timer.start(1000)
        QTimer.singleShot(0, edit)
        try:
            (opener if opener is not None else self.window._configure_commands)()
        finally:
            timer.stop()
        if failures:
            raise failures[0]

    def command_fields(self, dialog: QDialog) -> tuple[QComboBox, QLineEdit, QLineEdit]:
        channel_combo = dialog.findChild(QComboBox, "commandChannel")
        pages = dialog.findChild(QStackedWidget, "commandPages")
        assert channel_combo is not None and pages is not None
        page = pages.currentWidget()
        assert page is not None
        temperature = page.findChild(QLineEdit, "temperatureCommand")
        speed = page.findChild(QLineEdit, "speedCommand")
        assert temperature is not None and speed is not None
        return channel_combo, temperature, speed

    def test_bulk_sensor_commands_shows_and_saves_every_channel_without_execution(self) -> None:
        def save_all_commands(dialog: QDialog) -> None:
            table = dialog.findChild(QTableWidget, "allCommandChannels")
            assert table is not None
            self.assertEqual(table.rowCount(), 2)
            self.assertIsNone(dialog.findChild(QComboBox, "commandChannel"))
            for row in range(table.rowCount()):
                self.assertEqual(table.item(row, 0).text(), f"Channel {row + 1:02d}")
                temperature = table.cellWidget(row, 1)
                speed = table.cellWidget(row, 2)
                assert isinstance(temperature, QLineEdit) and isinstance(speed, QLineEdit)
                self.assertTrue(temperature.isVisible())
                self.assertTrue(speed.isVisible())
                temperature.setText(f"temperature-reader --channel {row + 1}")
                speed.setText(f"fan-reader --channel {row + 1}")
            buttons = dialog.findChild(QDialogButtonBox)
            assert buttons is not None
            buttons.button(QDialogButtonBox.StandardButton.Save).click()

        with patch("subprocess.Popen") as subprocess:
            self.edit_sensor_commands(save_all_commands, self.window.commands_button.click)
            subprocess.assert_not_called()
        prefix = "commands/devices/Test%20API/Stereo%20microphone/channels"
        for channel in (1, 2):
            self.assertEqual(self.settings.value(f"{prefix}/{channel}/temperature"), f"temperature-reader --channel {channel}")
            self.assertEqual(self.settings.value(f"{prefix}/{channel}/speed"), f"fan-reader --channel {channel}")

    def test_bulk_sensor_commands_cancel_discards_all_row_changes(self) -> None:
        prefix = "commands/devices/Test%20API/Stereo%20microphone/channels"
        for channel in (1, 2):
            self.settings.setValue(f"{prefix}/{channel}/temperature", f"temperature-{channel}")
            self.settings.setValue(f"{prefix}/{channel}/speed", f"speed-{channel}")

        def cancel_all_changes(dialog: QDialog) -> None:
            table = dialog.findChild(QTableWidget, "allCommandChannels")
            assert table is not None
            for row in range(table.rowCount()):
                temperature = table.cellWidget(row, 1)
                speed = table.cellWidget(row, 2)
                assert isinstance(temperature, QLineEdit) and isinstance(speed, QLineEdit)
                self.assertEqual(temperature.text(), f"temperature-{row + 1}")
                self.assertEqual(speed.text(), f"speed-{row + 1}")
                temperature.setText("unsaved temperature")
                speed.setText("unsaved speed")
            buttons = dialog.findChild(QDialogButtonBox)
            assert buttons is not None
            buttons.button(QDialogButtonBox.StandardButton.Cancel).click()

        self.edit_sensor_commands(cancel_all_changes, self.window.commands_button.click)
        for channel in (1, 2):
            self.assertEqual(self.settings.value(f"{prefix}/{channel}/temperature"), f"temperature-{channel}")
            self.assertEqual(self.settings.value(f"{prefix}/{channel}/speed"), f"speed-{channel}")

    def test_bulk_and_individual_command_editors_share_saved_configuration(self) -> None:
        def save_individual_commands(dialog: QDialog) -> None:
            channel_combo, temperature, speed = self.command_fields(dialog)
            self.assertEqual(channel_combo.currentData(), 1)
            temperature.setText("individual-temperature")
            speed.setText("individual-speed")
            dialog.accept()

        self.edit_sensor_commands(save_individual_commands, self.window.channel_widgets[1].commands_button.click)

        def update_bulk_commands(dialog: QDialog) -> None:
            table = dialog.findChild(QTableWidget, "allCommandChannels")
            assert table is not None
            temperature = table.cellWidget(1, 1)
            speed = table.cellWidget(1, 2)
            assert isinstance(temperature, QLineEdit) and isinstance(speed, QLineEdit)
            self.assertEqual(temperature.text(), "individual-temperature")
            self.assertEqual(speed.text(), "individual-speed")
            temperature.setText("bulk-temperature")
            speed.setText("bulk-speed")
            dialog.accept()

        self.edit_sensor_commands(update_bulk_commands, self.window.commands_button.click)

        def check_individual_commands(dialog: QDialog) -> None:
            channel_combo, temperature, speed = self.command_fields(dialog)
            self.assertEqual(channel_combo.currentData(), 1)
            self.assertEqual(temperature.text(), "bulk-temperature")
            self.assertEqual(speed.text(), "bulk-speed")
            dialog.reject()

        self.edit_sensor_commands(check_individual_commands, self.window.channel_widgets[1].commands_button.click)

    def test_sensor_commands_are_saved_per_channel_without_execution(self) -> None:
        def save_commands(dialog: QDialog) -> None:
            channel_combo, temperature, speed = self.command_fields(dialog)
            self.assertEqual(channel_combo.count(), 2)
            temperature.setText("temperature-reader --channel 1")
            speed.setText("fan-reader --channel 1")
            channel_combo.setCurrentIndex(1)
            _, temperature, speed = self.command_fields(dialog)
            self.assertEqual(temperature.text(), "")
            self.assertEqual(speed.text(), "")
            temperature.setText("temperature-reader --channel 2")
            speed.setText("fan-reader --channel 2")
            channel_combo.setCurrentIndex(0)
            _, temperature, speed = self.command_fields(dialog)
            self.assertEqual(temperature.text(), "temperature-reader --channel 1")
            self.assertEqual(speed.text(), "fan-reader --channel 1")
            buttons = dialog.findChild(QDialogButtonBox)
            assert buttons is not None
            buttons.button(QDialogButtonBox.StandardButton.Save).click()

        with patch("subprocess.Popen") as subprocess:
            self.edit_sensor_commands(save_commands)
            subprocess.assert_not_called()
        prefix = "commands/devices/Test%20API/Stereo%20microphone/channels"
        for channel in (1, 2):
            self.assertEqual(self.settings.value(f"{prefix}/{channel}/temperature"), f"temperature-reader --channel {channel}")
            self.assertEqual(self.settings.value(f"{prefix}/{channel}/speed"), f"fan-reader --channel {channel}")

    def test_channel_button_loads_matching_commands_and_cancel_discards_edits(self) -> None:
        prefix = "commands/devices/Test%20API/Stereo%20microphone/channels/2"
        self.settings.setValue(f"{prefix}/temperature", "temperature-reader --channel 2")
        self.settings.setValue(f"{prefix}/speed", "fan-reader --channel 2")

        def cancel_changes(dialog: QDialog) -> None:
            channel_combo, temperature, speed = self.command_fields(dialog)
            self.assertEqual(channel_combo.currentData(), 1)
            self.assertEqual(temperature.text(), "temperature-reader --channel 2")
            self.assertEqual(speed.text(), "fan-reader --channel 2")
            temperature.setText("unsaved temperature")
            speed.setText("unsaved speed")
            buttons = dialog.findChild(QDialogButtonBox)
            assert buttons is not None
            buttons.button(QDialogButtonBox.StandardButton.Cancel).click()

        self.edit_sensor_commands(cancel_changes, self.window.channel_widgets[1].commands_button.click)
        self.assertEqual(self.settings.value(f"{prefix}/temperature"), "temperature-reader --channel 2")
        self.assertEqual(self.settings.value(f"{prefix}/speed"), "fan-reader --channel 2")

    def test_sensor_commands_are_device_specific_and_ignore_device_index_changes(self) -> None:
        stereo_prefix = "commands/devices/Test%20API/Stereo%20microphone/channels/1"
        self.settings.setValue(f"{stereo_prefix}/temperature", "stereo-temperature")
        self.settings.setValue(f"{stereo_prefix}/speed", "stereo-speed")
        self.window.device_combo.setCurrentIndex(self.window.device_combo.findText(self.devices[1].label, Qt.MatchFlag.MatchEndsWith))

        def save_mono_commands(dialog: QDialog) -> None:
            channel_combo, temperature, speed = self.command_fields(dialog)
            self.assertEqual(channel_combo.count(), 1)
            self.assertEqual(temperature.text(), "")
            self.assertEqual(speed.text(), "")
            temperature.setText("mono-temperature")
            speed.setText("mono-speed")
            dialog.accept()

        self.edit_sensor_commands(save_mono_commands)
        mono_prefix = "commands/devices/Test%20API/Mono%20microphone/channels/1"
        self.assertEqual(self.settings.value(f"{mono_prefix}/temperature"), "mono-temperature")
        self.assertEqual(self.settings.value(f"{mono_prefix}/speed"), "mono-speed")
        self.query.return_value = [justfan.InputDevice(33, "Stereo microphone", "Test API", 2, 48000, 0.01), self.devices[1]]
        self.window.refresh_devices()
        self.window.device_combo.setCurrentIndex(self.window.device_combo.findText(self.devices[0].label, Qt.MatchFlag.MatchEndsWith))

        def check_stereo_commands(dialog: QDialog) -> None:
            _, temperature, speed = self.command_fields(dialog)
            self.assertEqual(temperature.text(), "stereo-temperature")
            self.assertEqual(speed.text(), "stereo-speed")
            dialog.reject()

        self.edit_sensor_commands(check_stereo_commands)

    def test_sensor_commands_survive_reducing_and_restoring_channel_count(self) -> None:
        prefix = "commands/devices/Test%20API/Stereo%20microphone/channels/2"
        self.settings.setValue(f"{prefix}/temperature", "second-temperature")
        self.settings.setValue(f"{prefix}/speed", "second-speed")
        self.window.channel_count.setValue(1)
        self.window.start_capture()

        def save_single_channel(dialog: QDialog) -> None:
            channel_combo, _, _ = self.command_fields(dialog)
            self.assertEqual(channel_combo.count(), 1)
            dialog.accept()

        self.edit_sensor_commands(save_single_channel)
        self.window.channel_count.setValue(2)
        self.window.start_capture()

        def check_second_channel(dialog: QDialog) -> None:
            _, temperature, speed = self.command_fields(dialog)
            self.assertEqual(temperature.text(), "second-temperature")
            self.assertEqual(speed.text(), "second-speed")
            dialog.reject()

        self.edit_sensor_commands(check_second_channel, self.window.channel_widgets[1].commands_button.click)

    def test_legacy_global_commands_migrate_only_to_channel_one_on_save(self) -> None:
        self.settings.setValue("commands/temperature", "legacy-temperature")
        self.settings.setValue("commands/speed", "legacy-speed")

        def migrate_commands(dialog: QDialog) -> None:
            channel_combo, temperature, speed = self.command_fields(dialog)
            self.assertEqual(temperature.text(), "legacy-temperature")
            self.assertEqual(speed.text(), "legacy-speed")
            channel_combo.setCurrentIndex(1)
            _, temperature, speed = self.command_fields(dialog)
            self.assertEqual(temperature.text(), "")
            self.assertEqual(speed.text(), "")
            dialog.accept()

        self.edit_sensor_commands(migrate_commands)
        prefix = "commands/devices/Test%20API/Stereo%20microphone/channels"
        self.assertEqual(self.settings.value(f"{prefix}/1/temperature"), "legacy-temperature")
        self.assertEqual(self.settings.value(f"{prefix}/1/speed"), "legacy-speed")
        self.assertEqual(self.settings.value(f"{prefix}/2/temperature"), "")
        self.assertEqual(self.settings.value(f"{prefix}/2/speed"), "")
        self.assertFalse(self.settings.contains("commands/temperature"))
        self.assertFalse(self.settings.contains("commands/speed"))


if __name__ == "__main__":
    unittest.main()