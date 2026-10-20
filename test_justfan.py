"""Deterministic signal and offscreen UI tests; no microphone is opened."""

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QLineEdit

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
        self.assertEqual(combo.currentData().index, 3)
        self.assertEqual(self.check.call_args.kwargs["device"], 3)
        combo.setCurrentIndex(combo.findText(devices[0].label))
        self.window.refresh_devices()
        self.assertEqual(combo.currentData().index, 12)
        self.assertEqual(self.check.call_args.kwargs["device"], 12)

    def test_device_switch_clears_readings_and_queries_parameters(self) -> None:
        self.push()
        previous = self.window.capture.stream
        assert previous is not None
        self.window.device_combo.setCurrentIndex(self.window.device_combo.findText(self.devices[1].label))
        self.assertTrue(previous.closed)
        self.assertEqual(len(self.window.channel_widgets), 1)
        self.assertEqual(len(self.window.spectra.curves), 1)
        self.assertEqual(self.window.channel_widgets[0].current_values[0].text(), "--")
        self.assertEqual(self.window.sample_rate.value(), 44100)
        self.assertEqual(self.window.channel_count.value(), 1)
        self.assertEqual(self.window.latency.value(), 20)
        self.assertIsNone(self.window.spectra.curves[0].getData()[0])

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
        self.assertEqual(self.window.channel_widgets, [])
        self.assertEqual(self.window.spectra.curves, [])
        self.query.return_value = self.devices
        self.window.refresh_devices()
        self.assertTrue(self.window.start_button.isEnabled())
        self.assertEqual(len(self.window.channel_widgets), 2)

    def test_disconnect_is_reported(self) -> None:
        stream = self.window.capture.stream
        assert stream is not None
        stream.active = False
        self.window._update_measurement()
        self.assertTrue(stream.closed)
        self.assertIn("stopped unexpectedly", self.window.notice.text())

    def test_sensor_commands_are_saved_without_execution(self) -> None:
        captured = []

        def save_commands() -> None:
            assert self.application is not None
            dialog = self.application.activeModalWidget()
            if not isinstance(dialog, QDialog):
                return
            fields = dialog.findChildren(QLineEdit)
            fields[0].setText("temperature-reader --value")
            fields[1].setText("fan-reader --rpm")
            captured.extend(field.text() for field in fields)
            buttons = dialog.findChild(QDialogButtonBox)
            assert buttons is not None
            buttons.button(QDialogButtonBox.StandardButton.Save).click()

        QTimer.singleShot(0, save_commands)
        QTimer.singleShot(1000, lambda: self.application.activeModalWidget().reject() if self.application is not None and isinstance(self.application.activeModalWidget(), QDialog) else None)
        with patch("subprocess.Popen") as subprocess:
            self.window._configure_commands()
            subprocess.assert_not_called()
        self.assertEqual(len(captured), 2)
        self.assertEqual(self.settings.value("commands/temperature"), captured[0])
        self.assertEqual(self.settings.value("commands/speed"), captured[1])


if __name__ == "__main__":
    unittest.main()