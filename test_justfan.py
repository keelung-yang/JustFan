"""Deterministic signal and offscreen UI tests; no microphone is opened."""

import os
import signal

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import runpy
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
from PySide6.QtCore import QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPixmapCache
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QDialogButtonBox, QGraphicsItem, QLabel, QLineEdit, QStackedWidget, QTableWidget, QWidget

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


class WaterfallHistoryTests(unittest.TestCase):
    def test_compression_bounds_memory_and_preserves_narrow_peaks_per_channel(self) -> None:
        spectrum = np.full((16385, 2), justfan.DB_FLOOR)
        spectrum[100, 0] = -12
        spectrum[-1, 1] = -24
        compressed = justfan.WaterfallHistory.compress(spectrum)
        self.assertLessEqual(len(compressed), justfan.WaterfallHistory.MAX_BINS)
        self.assertEqual(compressed.dtype, np.float32)
        np.testing.assert_array_equal(compressed.max(axis=0), [-12, -24])

    def test_rows_follow_audio_time_and_keep_channels_independent(self) -> None:
        history = justfan.WaterfallHistory(2, seconds=30)
        first = np.array([[-12, -24], [-36, -48]], dtype=np.float32)
        second = first - 6
        history.append(first, 100, 100, 1000)
        history.append(second, 200, 100, 1000)
        for channel in range(2):
            image = history.image(channel)
            assert image is not None
            np.testing.assert_array_equal(image[-2], first[:, channel])
            np.testing.assert_array_equal(image[-1], second[:, channel])
            np.testing.assert_array_equal(image[:-2], justfan.DB_FLOOR)

    def test_same_time_slice_holds_peaks_and_ignores_duplicate_frames(self) -> None:
        history = justfan.WaterfallHistory(1, seconds=30)
        history.append(np.array([[-12], [-48]]), 25, 25, 1000)
        history.append(np.array([[-24], [-36]]), 50, 25, 1000)
        history.append(np.zeros((2, 1)), 50, 25, 1000)
        image = history.image(0)
        assert image is not None
        np.testing.assert_array_equal(image[-1], [-12, -36])
        np.testing.assert_array_equal(image[:-1], justfan.DB_FLOOR)

    def test_missing_audio_time_is_not_stretched_into_continuous_data(self) -> None:
        history = justfan.WaterfallHistory(1, seconds=30)
        history.append(np.array([[-12]]), 100, 100, 1000)
        history.append(np.array([[-24]]), 400, 100, 1000)
        image = history.image(0)
        assert image is not None
        np.testing.assert_array_equal(image[-4:, 0], [-12, justfan.DB_FLOOR, justfan.DB_FLOOR, -24])

    def test_history_wraps_and_drops_data_older_than_the_window(self) -> None:
        history = justfan.WaterfallHistory(1, seconds=30)
        history.append(np.array([[-12]]), 100, 100, 1000)
        history.append(np.array([[-24]]), 30100, 100, 1000)
        image = history.image(0)
        assert image is not None and history.data is not None
        self.assertEqual(image.shape, (history.ROWS, 1))
        self.assertEqual(history.data.nbytes, history.ROWS * 4)
        np.testing.assert_array_equal(image[:-1], justfan.DB_FLOOR)
        np.testing.assert_array_equal(image[-1], [-24])

    def test_long_fft_blocks_fill_their_real_duration_without_mutating_inputs(self) -> None:
        history = justfan.WaterfallHistory(1, seconds=30)
        levels = np.array([[-18]], dtype=np.float32)
        history.append(levels, 500, 500, 1000)
        levels.fill(0)
        image = history.image(0)
        assert image is not None
        np.testing.assert_array_equal(image[-5:], -18)
        np.testing.assert_array_equal(image[:-5], justfan.DB_FLOOR)
        image.fill(0)
        fresh = history.image(0)
        assert fresh is not None
        np.testing.assert_array_equal(fresh[-5:], -18)


    def test_capture_retains_unrendered_spectra_without_consuming_snapshots(self) -> None:
        analyzer = justfan.SpectrumAnalyzer(48000, 256)
        buffer = justfan.CaptureBuffer(2, analyzer)
        first = np.column_stack((np.full(256, 0.5), np.zeros(256)))
        second = np.column_stack((np.zeros(256), np.full(256, 0.25)))
        buffer.push(first)
        buffer.push(second)
        frame = buffer.snapshot()
        assert frame is not None
        self.assertEqual([frames for frames, _ in frame.spectrum_history], [256, 512])
        for (_, levels), samples in zip(frame.spectrum_history, (first, second)):
            np.testing.assert_allclose(levels, analyzer.analyze(samples), atol=1e-5)
        again = buffer.snapshot()
        assert again is not None
        self.assertEqual([frames for frames, _ in again.spectrum_history], [256, 512])

    def test_capture_history_is_bounded_and_resets_with_measurement_time(self) -> None:
        analyzer = justfan.SpectrumAnalyzer(48000, 256)
        buffer = justfan.CaptureBuffer(1, analyzer)
        samples = np.full((256, 1), 0.25)
        for _block in range(buffer.MAX_HISTORY_BLOCKS + 5):
            buffer.push(samples)
        frame = buffer.snapshot()
        assert frame is not None
        self.assertEqual(len(frame.spectrum_history), buffer.MAX_HISTORY_BLOCKS)
        self.assertEqual(frame.spectrum_history[0][0], 6 * 256)
        buffer.reset_peaks()
        reset = buffer.snapshot()
        assert reset is not None
        self.assertEqual(len(reset.spectrum_history), 1)
        self.assertEqual(reset.spectrum_history[0][0], 0)
        np.testing.assert_allclose(reset.spectrum_history[0][1], analyzer.analyze(samples), atol=1e-5)


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


class RecordingPathTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def test_writable_current_directory_uses_records_and_removes_probes(self) -> None:
        with (
            patch.object(justfan.Path, "cwd", return_value=self.directory),
            patch.object(justfan.QStandardPaths, "writableLocation") as music,
        ):
            result = justfan.recording_directory()
        self.assertEqual(result, self.directory / "Records")
        self.assertTrue(result.is_dir())
        self.assertEqual(list(result.iterdir()), [])
        self.assertEqual(list(self.directory.iterdir()), [result])
        music.assert_not_called()

    def test_unwritable_current_or_records_directory_falls_back_to_music(self) -> None:
        music = self.directory / "Music"
        probe = justfan.TemporaryFile
        for denied in (self.directory, self.directory / "Records"):
            with self.subTest(denied=denied):
                def checked_probe(*, dir: Path):
                    if dir == denied:
                        raise PermissionError("Directory is read-only")
                    return probe(dir=dir)

                with (
                    patch.object(justfan.Path, "cwd", return_value=self.directory),
                    patch.object(justfan.QStandardPaths, "writableLocation", return_value=str(music)),
                    patch.object(justfan, "TemporaryFile", side_effect=checked_probe),
                ):
                    result = justfan.recording_directory()
                self.assertEqual(result, music / "JustFan")
                self.assertTrue(result.is_dir())
                self.assertEqual(list(result.iterdir()), [])

    def test_no_writable_directory_reports_failure(self) -> None:
        with (
            patch.object(justfan.Path, "cwd", return_value=self.directory),
            patch.object(justfan.QStandardPaths, "writableLocation", return_value=str(self.directory / "Music")),
            patch.object(justfan, "TemporaryFile", side_effect=PermissionError("No writable directory")),
        ):
            with self.assertRaisesRegex(PermissionError, "No writable directory"):
                justfan.recording_directory()

    def test_file_name_uses_device_item_and_safe_start_time(self) -> None:
        path = justfan.new_recording_path(self.directory, '1. Mic\u00ae\u00a9\u2122 / USB: "Input" [WASAPI]')
        self.assertRegex(path.name, r'^1\. Mic _ USB_ _Input_ \[WASAPI\]_\d{8}-\d{6}-\d{6}_recording\.wav$')
        self.assertEqual(path.parent, self.directory)
        self.assertEqual(path.stat().st_size, 0)
        localized = justfan.new_recording_path(self.directory, "2. \u9ea6\u514b\u98ce Caf\u00e9 + USB [API]")
        self.assertTrue(localized.name.startswith("2. \u9ea6\u514b\u98ce Caf\u00e9 + USB [API]_"))
        symbols_only = justfan.new_recording_path(self.directory, "\u00ae\u00a9\u2122")
        self.assertTrue(symbols_only.name.startswith("Device_"))
        long_name = justfan.new_recording_path(self.directory, "\u9ea6\u514b\u98ce" * 100)
        self.assertLess(len(long_name.name.encode("utf-8")), 255)

    def test_identical_start_times_never_overwrite_an_existing_file(self) -> None:
        started = justfan.datetime(2026, 10, 22, 12, 30, 45, 123456)
        with patch.object(justfan, "datetime") as clock:
            clock.now.return_value = started
            first = justfan.new_recording_path(self.directory, "1. Microphone [API]")
            first.write_bytes(b"Existing recording")
            second = justfan.new_recording_path(self.directory, "1. Microphone [API]")
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), b"Existing recording")
        self.assertTrue(second.exists())


class RecordingTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "recording.wav"
        self.settings = justfan.AudioSettings(48000, 2, 2048, 0.01)

    def test_float32_format_preserves_samples_and_channel_order(self) -> None:
        recorder = justfan.AudioRecorder(self.path, self.settings)
        self.addCleanup(recorder.stop)
        samples = np.array(
            [[-1.2, 1.2], [-1.0, 1.0], [-0.5, 0.5], [0.0, -0.0], [1e-10, -1e-10], [1e-30, -1e-30]],
            dtype=np.float32,
        )
        expected = samples.copy()
        recorder.push(samples)
        samples.fill(0)
        recorder.stop()
        self.assertEqual(recorder.error, "")
        with sf.SoundFile(self.path) as recording:
            self.assertEqual(recording.format, "WAV")
            self.assertEqual(recording.subtype, "FLOAT")
            self.assertEqual(recording.channels, 2)
            self.assertEqual(recording.samplerate, 48000)
            self.assertEqual(recording.frames, len(expected))
            actual = recording.read(dtype="float32", always_2d=True)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        self.assertEqual(self.path.stat().st_size, recorder._header_bytes + expected.nbytes)

    def test_all_blocks_and_eight_channels_are_saved(self) -> None:
        settings = justfan.AudioSettings(44100, 8, 2048, 0.01)
        recorder = justfan.AudioRecorder(self.path, settings)
        self.addCleanup(recorder.stop)
        samples = np.arange(32, dtype=np.float32).reshape(4, 8) / 32
        recorder.push(samples[:2])
        recorder.push(samples[2:])
        recorder.stop()
        recorder.stop()
        self.assertEqual(recorder.error, "")
        self.assertEqual(recorder.frames, 4)
        with sf.SoundFile(self.path) as recording:
            self.assertEqual(recording.channels, 8)
            self.assertEqual(recording.samplerate, 44100)
            self.assertEqual(recording.frames, 4)
            self.assertEqual(recording.subtype, "FLOAT")
            actual = recording.read(dtype="float32", always_2d=True)
        np.testing.assert_array_equal(actual, samples)
        self.assertEqual(self.path.stat().st_size, recorder._header_bytes + 4 * 8 * 4)

    def test_empty_recording_has_a_valid_header(self) -> None:
        recorder = justfan.AudioRecorder(self.path, self.settings)
        recorder.stop()
        self.assertEqual(recorder.error, "")
        with sf.SoundFile(self.path) as recording:
            self.assertEqual(recording.frames, 0)
            self.assertEqual(recording.channels, 2)
            self.assertEqual(recording.subtype, "FLOAT")

    def test_automatic_recording_name_uses_the_written_audio_duration(self) -> None:
        settings = justfan.AudioSettings(4, 2, 4, 0.01)
        for frames, duration in (
            (1, "000000"), (239, "000059"), (240, "000100"),
            (14399, "005959"), (14400, "010000"), (14895, "010203"),
        ):
            with self.subTest(frames=frames):
                path = justfan.new_recording_path(self.path.parent, "1. Microphone [API]")
                recorder = justfan.AudioRecorder(path, settings, append_duration=True)
                self.addCleanup(recorder.stop)
                recorder.push(np.zeros((frames, 2), dtype=np.float32))
                recorder.stop()
                self.assertEqual(recorder.error, "")
                self.assertFalse(path.exists())
                self.assertEqual(recorder.path.name, path.stem.removesuffix("_recording") + f"_{duration}.wav")
                with sf.SoundFile(recorder.path) as recording:
                    self.assertEqual(recording.frames, frames)
                recorder.stop()
                self.assertTrue(recorder.path.exists())

    def test_final_recording_name_never_overwrites_an_existing_file(self) -> None:
        path = justfan.new_recording_path(self.path.parent, "1. Microphone [API]")
        existing = path.with_name(path.stem.removesuffix("_recording") + "_000000.wav")
        existing.write_bytes(b"Existing recording")
        recorder = justfan.AudioRecorder(path, self.settings, append_duration=True)
        self.addCleanup(recorder.stop)
        recorder.stop()
        self.assertEqual(recorder.error, "")
        self.assertNotEqual(recorder.path, existing)
        self.assertEqual(existing.read_bytes(), b"Existing recording")
        with sf.SoundFile(recorder.path) as recording:
            self.assertEqual(recording.frames, 0)

    def test_failed_final_rename_retains_the_recording(self) -> None:
        path = justfan.new_recording_path(self.path.parent, "1. Microphone [API]")
        recorder = justfan.AudioRecorder(path, self.settings, append_duration=True)
        self.addCleanup(recorder.stop)
        recorder.push(np.zeros((16, 2), dtype=np.float32))
        with patch.object(justfan.Path, "replace", side_effect=PermissionError("Rename denied")):
            recorder.stop()
        self.assertIn("Rename denied", recorder.error)
        self.assertEqual(recorder.path, path)
        with sf.SoundFile(path) as recording:
            self.assertEqual(recording.frames, 16)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_disk_failure_is_reported_without_raising_in_callback(self) -> None:
        recorder = justfan.AudioRecorder(self.path, self.settings)
        self.addCleanup(recorder.stop)
        with patch.object(recorder._file, "write", side_effect=OSError("Disk full")):
            recorder.push(np.zeros((16, 2), dtype=np.float32))
            recorder.stop()
        self.assertIn("Disk full", recorder.error)

    def test_full_queue_stops_recording_without_blocking_the_callback(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        with patch.object(justfan.AudioRecorder, "QUEUE_SIZE", 1):
            recorder = justfan.AudioRecorder(self.path, self.settings)
        self.addCleanup(recorder.stop)
        self.addCleanup(release.set)
        write = recorder._file.write

        def slow_write(samples: np.ndarray) -> None:
            entered.set()
            if not release.wait(2.0):
                raise TimeoutError("Test writer was not released")
            write(samples)

        with patch.object(recorder._file, "write", side_effect=slow_write):
            samples = np.zeros((16, 2), dtype=np.float32)
            recorder.push(samples)
            self.assertTrue(entered.wait(2.0))
            recorder.push(samples)
            recorder.push(samples)
            self.assertIn("could not keep up", recorder.error)
            release.set()
            recorder.stop()
        with sf.SoundFile(self.path) as recording:
            self.assertEqual(recording.frames, 32)

    def test_invalid_recording_settings_do_not_create_a_file(self) -> None:
        for settings in (
            justfan.AudioSettings(0, 2, 2048, 0.01),
            justfan.AudioSettings(48000.1, 2, 2048, 0.01),
            justfan.AudioSettings(48000, 0, 2048, 0.01),
        ):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    justfan.AudioRecorder(self.path, settings)
                self.assertFalse(self.path.exists())

    def test_wav_size_limit_preserves_a_readable_partial_recording(self) -> None:
        recorder = justfan.AudioRecorder(self.path, self.settings)
        self.addCleanup(recorder.stop)
        samples = np.zeros((16, 2), dtype=np.float32)
        limit = recorder._header_bytes + samples.nbytes
        with patch.object(recorder, "MAX_FILE_BYTES", limit):
            recorder.push(samples)
            recorder.push(samples)
            recorder.stop()
        self.assertIn("4 GiB", recorder.error)
        self.assertEqual(self.path.stat().st_size, limit)
        with sf.SoundFile(self.path) as recording:
            self.assertEqual(recording.frames, 16)


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

    def test_recording_saves_all_callbacks_only_between_start_and_stop(self) -> None:
        capture = justfan.AudioCapture()
        self.addCleanup(capture.stop)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.wav"
            with patch.object(justfan.sd, "check_input_settings"), patch.object(justfan.sd, "InputStream", side_effect=FakeStream):
                capture.start(self.device, self.settings)
            stream = capture.stream
            assert isinstance(stream, FakeStream)
            samples = np.full((2048, 2), 0.25, dtype=np.float32)
            stream.callback(samples, 2048, None, "")
            capture.start_recording(path)
            stream.callback(samples, 2048, None, "")
            stream.callback(samples, 2048, None, "")
            recorder = capture.stop_recording()
            stream.callback(samples, 2048, None, "")
            self.assertIsNotNone(recorder)
            self.assertIsNone(capture.recorder)
            self.assertTrue(stream.active)
            with sf.SoundFile(path) as recording:
                self.assertEqual(recording.frames, 4096)
                self.assertEqual(recording.channels, 2)

    def test_recording_requires_capture_and_rejects_duplicate_start(self) -> None:
        capture = justfan.AudioCapture()
        self.addCleanup(capture.stop)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.wav"
            with self.assertRaisesRegex(ValueError, "Start audio capture"):
                capture.start_recording(path)
            self.assertFalse(path.exists())
            with patch.object(justfan.sd, "check_input_settings"), patch.object(justfan.sd, "InputStream", side_effect=FakeStream):
                capture.start(self.device, self.settings)
            capture.start_recording(path)
            recorder = capture.recorder
            with self.assertRaisesRegex(ValueError, "already in progress"):
                capture.start_recording(path)
            self.assertIs(capture.recorder, recorder)
            capture.stop()
            self.assertIsNone(capture.recorder)
            with sf.SoundFile(path) as recording:
                self.assertEqual(recording.frames, 0)

    def test_late_callback_after_restart_cannot_enter_a_new_recording(self) -> None:
        capture = justfan.AudioCapture()
        self.addCleanup(capture.stop)
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.wav"
            second_path = Path(directory) / "second.wav"
            with patch.object(justfan.sd, "check_input_settings"), patch.object(justfan.sd, "InputStream", side_effect=FakeStream):
                capture.start(self.device, self.settings)
                old_stream = capture.stream
                assert isinstance(old_stream, FakeStream)
                capture.start_recording(first_path)
                samples = np.full((2048, 2), 0.25, dtype=np.float32)
                old_stream.callback(samples, 2048, None, "")
                capture.start(self.device, self.settings)
            stream = capture.stream
            assert isinstance(stream, FakeStream)
            capture.start_recording(second_path)
            old_stream.callback(samples, 2048, None, "")
            stream.callback(samples, 2048, None, "")
            capture.stop()
            for path in (first_path, second_path):
                with sf.SoundFile(path) as recording:
                    self.assertEqual(recording.frames, 2048)

    def test_stop_closes_even_if_driver_stop_fails(self) -> None:
        capture = justfan.AudioCapture()
        stream = FakeStream(callback=lambda *_args: None)
        capture.stream = stream
        with patch.object(stream, "stop", side_effect=justfan.sd.PortAudioError("Disconnected")):
            capture.stop()
        self.assertTrue(stream.closed)
        self.assertIsNone(capture.stream)

    def test_stop_does_not_block_indefinitely_when_driver_hangs(self) -> None:
        capture = justfan.AudioCapture()
        stream = FakeStream(callback=lambda *_args: None)
        capture.stream = stream
        release = threading.Event()
        self.addCleanup(release.set)
        with patch.object(capture, "STOP_TIMEOUT", 0.02), patch.object(stream, "stop", side_effect=lambda: release.wait(1.0)):
            capture.stop()
        self.assertIsNone(capture.stream)
        release.set()


class PackagingTests(unittest.TestCase):
    def test_runtime_window_icon_is_included_in_bundle(self) -> None:
        project_root = Path(justfan.__file__).resolve().parent
        build_analysis = Mock(return_value=SimpleNamespace(pure=[], scripts=[], binaries=[], datas=[]))
        build_executable = Mock()
        runpy.run_path(
            str(project_root / "JustFan.spec"),
            init_globals={
                "SPECPATH": str(project_root),
                "Analysis": build_analysis,
                "PYZ": Mock(),
                "EXE": build_executable,
            },
        )
        icon_path = str(project_root / "images" / "JustFan.ico")
        self.assertIn((icon_path, "images"), build_analysis.call_args.kwargs.get("datas", []))
        self.assertIn((str(project_root / "images" / "JustFan.svg"), "images"), build_analysis.call_args.kwargs.get("datas", []))
        self.assertEqual(build_executable.call_args.kwargs["icon"], icon_path)


class LevelMeterTests(unittest.TestCase):
    application = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])
        assert isinstance(cls.application, QApplication)
        justfan.configure_application(cls.application)

    def setUp(self) -> None:
        self.meter = justfan.LevelMeter(justfan.CHANNEL_COLORS[2])
        self.meter.resize(240, 40)
        self.addCleanup(self.meter.close)

    def render_meter(self) -> QImage:
        image = QImage(self.meter.size(), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        self.meter.render(image)
        return image

    def test_empty_and_silent_meter(self) -> None:
        background = self.meter.palette().color(justfan.QPalette.ColorRole.AlternateBase)
        marker = self.meter.palette().color(justfan.QPalette.ColorRole.WindowText)
        row = self.meter.height() - 6
        image = self.render_meter()
        self.assertIsNone(self.meter.current)
        self.assertIsNone(self.meter.peak)
        for position in (0, 120, 239):
            self.assertEqual(image.pixelColor(position, row), background)
        self.meter.set_levels(justfan.DB_FLOOR, justfan.DB_FLOOR)
        image = self.render_meter()
        self.assertEqual(image.pixelColor(0, row), marker)
        self.assertEqual(image.pixelColor(2, row), background)

    def test_current_fill_and_peak_marker_share_dbfs_scale(self) -> None:
        fill = QColor(justfan.CHANNEL_COLORS[2])
        background = self.meter.palette().color(justfan.QPalette.ColorRole.AlternateBase)
        marker = self.meter.palette().color(justfan.QPalette.ColorRole.WindowText)
        row = self.meter.height() - 6
        for current, endpoint in ((-60.0, 120), (-90.0, 60)):
            with self.subTest(current=current):
                self.meter.set_levels(current, -30.0)
                image = self.render_meter()
                self.assertEqual(image.pixelColor(endpoint - 1, row), fill)
                self.assertEqual(image.pixelColor(endpoint, row), background)
                self.assertEqual(image.pixelColor(179, row), marker)
                self.assertEqual(image.pixelColor(180, row), marker)
                self.assertEqual(image.pixelColor(181, row), background)

    def test_display_clamps_at_both_edges_and_resizes(self) -> None:
        fill = QColor(justfan.CHANNEL_COLORS[2])
        background = self.meter.palette().color(justfan.QPalette.ColorRole.AlternateBase)
        marker = self.meter.palette().color(justfan.QPalette.ColorRole.WindowText)
        for width in (96, 480):
            with self.subTest(width=width):
                self.meter.resize(width, 40)
                row = self.meter.height() - 6
                self.meter.set_levels(-180.0, -150.0)
                image = self.render_meter()
                self.assertEqual(image.pixelColor(0, row), marker)
                self.assertEqual(image.pixelColor(2, row), background)
                self.assertEqual(self.meter.current, -180.0)
                self.meter.set_levels(6.0, 12.0)
                image = self.render_meter()
                self.assertEqual(image.pixelColor(0, row), fill)
                self.assertEqual(image.pixelColor(width - 3, row), fill)
                self.assertEqual(image.pixelColor(width - 1, row), marker)
                self.assertEqual(self.meter.peak, 12.0)

    def test_channel_readings_stay_inside_meter_at_compact_widths(self) -> None:
        channel = justfan.ChannelWidget(2)
        self.addCleanup(channel.close)
        channel.update_levels(justfan.DB_FLOOR, justfan.DB_FLOOR)
        channel.show()
        assert self.application is not None
        for width in (264, 467):
            with self.subTest(width=width):
                channel.resize(max(width, channel.minimumSizeHint().width()), channel.minimumSizeHint().height())
                self.application.processEvents()
                meter = channel.level_meter
                current = channel.current_values[0]
                peak = channel.peak_values[0]
                for label in (current, peak):
                    self.assertTrue(meter.rect().contains(label.geometry()))
                    self.assertGreaterEqual(label.y(), 6)
                    self.assertLess(label.geometry().bottom(), meter.height() - 6)
                    self.assertLessEqual(abs(label.geometry().center().y() - meter.rect().center().y()), 1)
                    self.assertGreaterEqual(label.width(), label.fontMetrics().horizontalAdvance(label.text()))
                    self.assertGreaterEqual(label.height(), label.minimumSizeHint().height())
                self.assertLess(current.geometry().right(), peak.geometry().left())

    def test_text_contrast_for_all_channel_colors_and_light_dark_backgrounds(self) -> None:
        for color in (*justfan.CHANNEL_COLORS, "#000000", "#ffffff", "#edf2f3", "#26343b"):
            with self.subTest(color=color):
                background = QColor(color)
                foreground = self.meter._text_color(background)
                luminances = []
                for sample in (background, foreground):
                    components = [
                        component / 12.92 if component <= 0.04045 else ((component + 0.055) / 1.055) ** 2.4
                        for component in (sample.redF(), sample.greenF(), sample.blueF())
                    ]
                    luminances.append(0.2126 * components[0] + 0.7152 * components[1] + 0.0722 * components[2])
                self.assertGreaterEqual((max(luminances) + 0.05) / (min(luminances) + 0.05), 4.5)

    def test_inline_text_changes_color_across_fill_and_peak_marker(self) -> None:
        channel = justfan.ChannelWidget(2)
        self.addCleanup(channel.close)
        channel.show()
        channel.resize(max(467, channel.minimumSizeHint().width()), channel.minimumSizeHint().height())
        assert self.application is not None
        self.application.processEvents()
        self.meter = channel.level_meter
        current_label = channel.current_values[0]
        peak_label = channel.peak_values[0]
        current_position = current_label.x() + current_label.fontMetrics().horizontalAdvance("-100.0") / 2
        peak_position = peak_label.geometry().right() - peak_label.fontMetrics().horizontalAdvance("-10.0") / 2
        current = justfan.DB_FLOOR + 120 * current_position / self.meter.width()
        peak = justfan.DB_FLOOR + 120 * peak_position / self.meter.width()
        for fill_color, track_color in (("#5364ab", "#edf2f3"), ("#f3dc73", "#18323b")):
            with self.subTest(fill=fill_color, track=track_color):
                self.meter._color = QColor(fill_color)
                palette = self.meter.palette()
                palette.setColor(justfan.QPalette.ColorRole.AlternateBase, QColor(track_color))
                self.meter.setPalette(palette)
                channel.update_levels(current, peak)
                image = self.render_meter()
                mask = QImage(self.meter.size(), QImage.Format.Format_ARGB32)
                mask.fill(Qt.GlobalColor.transparent)
                painter = QPainter(mask)
                try:
                    painter.setPen(Qt.GlobalColor.white)
                    for label in (current_label, peak_label):
                        painter.setFont(label.font())
                        painter.drawText(label.contentsRect().translated(label.pos()), label.alignment(), label.text())
                finally:
                    painter.end()
                backgrounds = set()
                for row in range(mask.height()):
                    for column in range(mask.width()):
                        if mask.pixelColor(column, row).alpha() == 255:
                            background = image.pixelColor(column, 0)
                            backgrounds.add(background.name())
                            self.assertEqual(image.pixelColor(column, row), self.meter._text_color(background))
                self.assertIn(fill_color, backgrounds)
                self.assertIn(track_color, backgrounds)


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
        self.recording_folder = Path(self.directory.name) / "Records"
        self.recording_folder.mkdir()
        self.destination = self.enterContext(patch.object(justfan, "recording_directory", return_value=self.recording_folder))
        self.window = justfan.MainWindow(self.settings)
        self.addCleanup(self.window.close)
        self.window.show()
        assert self.application is not None
        self.application.processEvents()
        self.window.start_capture()

    def test_initialization_does_not_start_capture(self) -> None:
        with patch.object(justfan.AudioCapture, "start") as start:
            window = justfan.MainWindow(self.settings)
        self.addCleanup(window.close)
        start.assert_not_called()
        self.assertIsNone(window.capture.stream)
        self.assertEqual(window.start_button.text(), "Start")
        self.assertEqual(len(window.channel_widgets), 2)
        self.assertFalse(window.record_checkbox.isChecked())
        self.assertTrue(window.record_checkbox.isEnabled())
        self.destination.assert_not_called()

    def test_ctrl_c_closes_window_and_restores_signal_handler(self) -> None:
        assert self.application is not None
        previous_handler = signal.getsignal(signal.SIGINT)
        QTimer.singleShot(50, lambda: signal.raise_signal(signal.SIGINT))
        timeout = QTimer(self.window)
        timeout.setSingleShot(True)
        timeout.timeout.connect(self.application.quit)
        timeout.start(2000)
        with (
            patch.object(justfan, "QApplication", return_value=self.application),
            patch.object(justfan, "configure_application"),
            patch.object(justfan, "MainWindow", return_value=self.window),
        ):
            self.assertEqual(justfan.main(), 0)
        timeout.stop()
        self.assertFalse(self.window.isVisible())
        self.assertIsNone(self.window.capture.stream)
        self.assertIs(signal.getsignal(signal.SIGINT), previous_handler)

    def test_windows_inherit_application_icon(self) -> None:
        assert self.application is not None
        icon = self.application.windowIcon()
        self.assertFalse(icon.isNull())
        self.assertEqual(self.window.windowIcon().cacheKey(), icon.cacheKey())
        dialog = QDialog(self.window)
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.windowIcon().cacheKey(), icon.cacheKey())
        sizes = {(size.width(), size.height()) for size in icon.availableSizes()}
        for size in (16, 24, 32, 48, 64, 128, 256):
            with self.subTest(size=size):
                self.assertIn((size, size), sizes)
                self.assertFalse(self.window.windowIcon().pixmap(size, size).isNull())

    def test_application_icon_loads_from_bundle_outside_working_directory(self) -> None:
        assert self.application is not None
        bundle = Path(self.directory.name) / "bundle"
        images = bundle / "images"
        images.mkdir(parents=True)
        for filename in ("JustFan.ico", "JustFan.svg"):
            (images / filename).write_bytes((Path(justfan.__file__).parent / "images" / filename).read_bytes())
        self.addCleanup(justfan.configure_application, self.application)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.directory.name)
        with patch.object(justfan, "__file__", str(bundle / "JustFan.py")):
            justfan.configure_application(self.application)
            self.assertFalse(self.application.windowIcon().isNull())
            self.assertFalse(self.window.windowIcon().pixmap(24, 24).isNull())
            animator = justfan.FanIconAnimator(self.window)
            self.assertFalse(animator.idle_icon.isNull())
            self.assertEqual(len(animator.frames), animator.POSE_FRAMES + animator.SPIN_FRAMES)

    def test_icon_straightens_rotates_and_restores_idle_on_stop(self) -> None:
        assert self.application is not None
        animator = self.window.icon_animator
        frames = animator.frames
        self.assertTrue(animator.timer.isActive())
        self.window._stop_capture()
        self.assertFalse(animator.timer.isActive())
        self.assertEqual(self.window.windowIcon().cacheKey(), animator.idle_icon.cacheKey())
        self.window.start_capture()
        self.assertTrue(animator.timer.isActive())
        self.assertIs(animator.frames, frames)
        images = []
        for _step in range(animator.POSE_FRAMES + 3):
            images.append(self.window.windowIcon().pixmap(QSize(32, 32), 1.0).toImage())
            animator._advance()
        self.assertNotEqual(images[0], images[animator.POSE_FRAMES - 1])
        self.assertNotEqual(images[animator.POSE_FRAMES], images[animator.POSE_FRAMES + 1])
        for _step in range(len(frames) * 2):
            animator._advance()
        self.assertGreaterEqual(animator.frame_index, animator.POSE_FRAMES)
        self.assertLess(animator.frame_index, len(frames))
        self.window._stop_capture()
        self.assertEqual(self.application.windowIcon().cacheKey(), animator.idle_icon.cacheKey())
        animator._advance()
        self.assertEqual(self.application.windowIcon().cacheKey(), animator.idle_icon.cacheKey())

    def test_running_icon_rotates_only_blades(self) -> None:
        animator = self.window.icon_animator
        images = [
            icon.pixmap(QSize(256, 256), 1.0).toImage().convertToFormat(QImage.Format.Format_RGBA8888)
            for icon in animator.frames[animator.POSE_FRAMES:animator.POSE_FRAMES + 2]
        ]
        pixels = [np.frombuffer(image.constBits(), dtype=np.uint8).copy().reshape(256, 256, 4) for image in images]
        changed = np.any(pixels[0] != pixels[1], axis=2)
        rows, columns = np.ogrid[:256, :256]
        radius = np.hypot((columns + 0.5 - 4.8) / 0.92 - 120, (rows + 0.5 - 16) / 0.92 - 120)
        self.assertTrue(np.any(changed))
        self.assertFalse(np.any(changed[radius > 82]))
        self.assertFalse(np.any(changed[radius < 22]))

    def test_icon_animation_stops_after_failure_disconnect_and_close(self) -> None:
        animator = self.window.icon_animator
        self.check.side_effect = justfan.sd.PortAudioError("Unsupported sample rate")
        self.window.start_capture()
        self.assertFalse(animator.timer.isActive())
        self.assertEqual(self.window.windowIcon().cacheKey(), animator.idle_icon.cacheKey())
        self.check.side_effect = None
        self.window.start_capture()
        self.assertTrue(animator.timer.isActive())
        stream = self.window.capture.stream
        assert stream is not None
        stream.active = False
        self.window._update_measurement()
        self.assertFalse(animator.timer.isActive())
        self.assertEqual(self.window.windowIcon().cacheKey(), animator.idle_icon.cacheKey())
        self.window.start_capture()
        self.assertTrue(animator.timer.isActive())
        self.window.close()
        self.assertFalse(animator.timer.isActive())
        self.assertEqual(self.window.windowIcon().cacheKey(), animator.idle_icon.cacheKey())

    def push(self, amplitude: float = 0.25) -> None:
        settings = self.window.capture.settings
        stream = self.window.capture.stream
        assert settings is not None and stream is not None
        stream.callback(np.full((settings.block_size, settings.channels), amplitude), settings.block_size, None, "")
        self.window._update_measurement()

    def record(self) -> justfan.AudioRecorder:
        self.window._stop_capture()
        self.window.record_checkbox.setChecked(True)
        self.window.start_capture()
        recorder = self.window.capture.recorder
        assert recorder is not None
        return recorder

    def test_recording_checkbox_saves_wav_until_capture_stops(self) -> None:
        self.push()
        recorder = self.record()
        original_path = recorder.path
        self.assertTrue(self.window.record_checkbox.isChecked())
        self.assertFalse(self.window.record_checkbox.isEnabled())
        self.assertEqual(self.window.statusBar().currentMessage(), "")
        self.assertFalse(self.window.state_label.isHidden())
        self.assertTrue(self.window.state_label.text().startswith("Live | "))
        self.assertIsNone(self.window.findChild(justfan.QLabel, "recording"))
        self.assertIn(str(original_path), self.window.record_checkbox.toolTip())
        self.assertEqual(original_path.parent, self.recording_folder)
        self.assertTrue(original_path.name.startswith(self.window.device_combo.currentText() + "_"))
        self.push(0.5)
        self.push(0.25)
        settings = self.window.capture.settings
        assert settings is not None
        self.window.start_button.click()
        self.assertIsNone(self.window.capture.recorder)
        self.assertIsNone(self.window.capture.stream)
        self.assertTrue(self.window.record_checkbox.isChecked())
        self.assertTrue(self.window.record_checkbox.isEnabled())
        self.assertEqual(self.window.statusBar().currentMessage(), "")
        self.assertFalse(self.window.state_label.isHidden())
        self.assertEqual(self.window.state_label.text(), "Stopped")
        self.assertIn(str(recorder.path), self.window.record_checkbox.toolTip())
        self.assertFalse(original_path.exists())
        self.assertRegex(recorder.path.name, r'_\d{8}-\d{6}-\d{6}_\d{6}\.wav$')
        with sf.SoundFile(recorder.path) as recording:
            self.assertEqual(recording.frames, settings.block_size * 2)
            self.assertEqual(recording.channels, settings.channels)
            self.assertEqual(recording.subtype, "FLOAT")
            self.assertEqual(recording.samplerate, int(settings.sample_rate))

    def test_recording_directory_and_open_failures_do_not_prevent_capture(self) -> None:
        self.window._stop_capture()
        self.destination.side_effect = PermissionError("No writable directory")
        self.window.record_checkbox.setChecked(True)
        self.assertIsNone(self.window.capture.recorder)
        self.assertFalse(self.window.record_checkbox.isChecked())
        self.assertTrue(self.window.record_checkbox.isEnabled())
        self.assertIn("No writable directory", self.window.notice.text())
        self.window.start_button.click()
        self.assertIsNotNone(self.window.capture.stream)
        self.assertFalse(self.window.record_checkbox.isEnabled())
        self.destination.side_effect = None
        for error in (OSError("Read-only folder"), sf.SoundFileError("Read-only folder")):
            with self.subTest(error=type(error).__name__):
                self.window._stop_capture()
                self.window.record_checkbox.setChecked(True)
                with patch.object(justfan, "AudioRecorder", side_effect=error):
                    self.window.start_button.click()
                self.assertEqual(list(self.recording_folder.iterdir()), [])
                self.assertIsNone(self.window.capture.recorder)
                self.assertFalse(self.window.record_checkbox.isChecked())
                self.assertFalse(self.window.record_checkbox.isEnabled())
                self.assertIsNotNone(self.window.capture.stream)
                self.assertIn("Read-only folder", self.window.notice.text())

    def test_recording_checkbox_checks_destination_while_idle_and_follows_start_stop(self) -> None:
        self.assertFalse(self.window.record_checkbox.isChecked())
        self.assertFalse(self.window.record_checkbox.isEnabled())
        palette = self.window.record_checkbox.palette()
        for role in (justfan.QPalette.ColorRole.WindowText, justfan.QPalette.ColorRole.Text, justfan.QPalette.ColorRole.ButtonText):
            self.assertNotEqual(
                palette.color(justfan.QPalette.ColorGroup.Active, role),
                palette.color(justfan.QPalette.ColorGroup.Disabled, role),
            )
        self.window.record_checkbox.click()
        self.assertFalse(self.window.record_checkbox.isChecked())
        self.assertIsNone(self.window.capture.recorder)
        self.destination.assert_not_called()
        self.window.start_button.click()
        self.assertTrue(self.window.record_checkbox.isEnabled())
        self.window.record_checkbox.setChecked(True)
        self.destination.assert_called_once_with()
        self.assertIsNone(self.window.capture.recorder)
        self.assertEqual(list(self.recording_folder.iterdir()), [])
        self.window.start_button.click()
        recorder = self.window.capture.recorder
        assert recorder is not None
        self.assertFalse(self.window.record_checkbox.isEnabled())
        self.window.record_checkbox.click()
        self.assertTrue(self.window.record_checkbox.isChecked())
        self.assertIs(self.window.capture.recorder, recorder)
        self.push()
        self.window.start_button.click()
        self.assertTrue(self.window.record_checkbox.isChecked())
        self.assertTrue(self.window.record_checkbox.isEnabled())
        self.assertTrue(recorder.path.exists())
        self.assertIsNone(self.window.capture.recorder)
        self.window.start_button.click()
        self.assertFalse(self.window.record_checkbox.isEnabled())
        self.assertIsNotNone(self.window.capture.recorder)
        self.assertIsNot(self.window.capture.recorder, recorder)

    def test_recording_destination_is_rechecked_when_capture_starts(self) -> None:
        self.window._stop_capture()
        self.window.record_checkbox.setChecked(True)
        self.destination.side_effect = PermissionError("Directory is now read-only")
        self.window.start_button.click()
        self.assertIsNotNone(self.window.capture.stream)
        self.assertIsNone(self.window.capture.recorder)
        self.assertFalse(self.window.record_checkbox.isChecked())
        self.assertFalse(self.window.record_checkbox.isEnabled())
        self.assertIn("Directory is now read-only", self.window.notice.text())
        self.assertEqual(list(self.recording_folder.iterdir()), [])

    def test_failed_capture_start_does_not_create_a_recording(self) -> None:
        self.window._stop_capture()
        self.window.record_checkbox.setChecked(True)
        self.check.side_effect = justfan.sd.PortAudioError("Unsupported sample rate")
        self.window.start_button.click()
        self.assertIsNone(self.window.capture.stream)
        self.assertIsNone(self.window.capture.recorder)
        self.assertTrue(self.window.record_checkbox.isChecked())
        self.assertTrue(self.window.record_checkbox.isEnabled())
        self.assertEqual(list(self.recording_folder.iterdir()), [])

    def test_recording_is_finalized_on_all_capture_exit_paths(self) -> None:
        for action in ("stop", "apply", "refresh", "device", "disconnect", "close"):
            with self.subTest(action=action):
                recorder = self.record()
                self.push()
                settings = self.window.capture.settings
                assert settings is not None
                if action == "stop":
                    self.window.start_button.click()
                elif action == "apply":
                    self.window.apply_button.click()
                elif action == "refresh":
                    self.window.refresh_devices()
                elif action == "device":
                    self.window.device_combo.setCurrentIndex(1 - self.window.device_combo.currentIndex())
                elif action == "disconnect":
                    stream = self.window.capture.stream
                    assert isinstance(stream, FakeStream)
                    stream.stop()
                    self.window._update_measurement()
                else:
                    self.window.close()
                if action == "apply":
                    self.assertIsNotNone(self.window.capture.recorder)
                    self.assertIsNot(self.window.capture.recorder, recorder)
                    self.assertFalse(self.window.record_checkbox.isEnabled())
                else:
                    self.assertIsNone(self.window.capture.recorder)
                    self.assertTrue(self.window.record_checkbox.isEnabled())
                self.assertEqual(self.window.statusBar().currentMessage(), "")
                with sf.SoundFile(recorder.path) as recording:
                    self.assertEqual(recording.frames, settings.block_size)
                    self.assertEqual(recording.channels, settings.channels)
                    self.assertEqual(recording.subtype, "FLOAT")

    def test_recording_ignores_channel_visibility_and_peak_resets(self) -> None:
        recorder = self.record()
        self.window.channel_widgets[0].enabled_checkbox.setChecked(False)
        self.push()
        self.window.reset_button.click()
        self.assertIsNotNone(self.window.capture.recorder)
        self.assertEqual(self.window.statusBar().currentMessage(), "")
        self.push()
        settings = self.window.capture.settings
        assert settings is not None
        self.window.start_button.click()
        with sf.SoundFile(recorder.path) as recording:
            self.assertEqual(recording.channels, 2)
            self.assertEqual(recording.frames, settings.block_size * 2)

    def test_recording_write_failure_stops_only_recording(self) -> None:
        self.record()
        recorder = self.window.capture.recorder
        assert recorder is not None
        with patch.object(recorder._file, "write", side_effect=OSError("Disk full")):
            self.push()
            recorder.stop()
        self.window._update_measurement()
        self.assertIsNone(self.window.capture.recorder)
        self.assertIsNotNone(self.window.capture.stream)
        self.assertFalse(self.window.record_checkbox.isEnabled())
        self.assertFalse(self.window.record_checkbox.isChecked())
        self.assertEqual(self.window.statusBar().currentMessage(), "")
        self.assertFalse(self.window.state_label.isHidden())
        self.assertTrue(self.window.state_label.text().startswith("Live | "))
        self.assertIn("Disk full", self.window.notice.text())
        self.window.start_button.click()
        self.assertTrue(self.window.record_checkbox.isEnabled())

    def test_recording_toolbar_stays_unclipped_and_stable_in_compact_windows(self) -> None:
        assert self.application is not None
        controls = self.window.findChild(QWidget, "controls")
        assert controls is not None
        self.record()
        for width in (860, 1360):
            with self.subTest(width=width):
                self.window.resize(width, 680)
                self.application.processEvents()
                widgets = (
                    self.window.device_combo, self.window.commands_button, self.window.record_checkbox,
                    self.window.start_button, self.window.apply_button, self.window.reset_button,
                )
                for index, widget in enumerate(widgets):
                    self.assertTrue(controls.rect().contains(widget.geometry()))
                    for other in widgets[index + 1:]:
                        self.assertFalse(widget.geometry().intersects(other.geometry()))
                self.assertGreaterEqual(self.window.record_checkbox.width(), self.window.record_checkbox.sizeHint().width())
        active_geometry = self.window.start_button.geometry()
        self.window.start_button.click()
        self.application.processEvents()
        self.assertEqual(self.window.start_button.geometry(), active_geometry)

    def test_recording_requires_running_capture_and_uses_active_settings(self) -> None:
        self.window._stop_capture()
        self.assertTrue(self.window.record_checkbox.isEnabled())
        recorder = self.record()
        self.assertFalse(self.window.record_checkbox.isEnabled())
        self.window.sample_rate.setValue(44100)
        self.window.channel_count.setValue(1)
        self.push()
        self.window.start_button.click()
        with sf.SoundFile(recorder.path) as recording:
            self.assertEqual(recording.samplerate, 48000)
            self.assertEqual(recording.channels, 2)

    def test_waterfall_switch_preserves_capture_and_spectrum_peaks(self) -> None:
        self.push(0.5)
        self.push(0.25)
        stream = self.window.capture.stream
        panel = self.window.spectra
        self.assertEqual(self.window.view_tabs.currentIndex(), 0)
        self.assertTrue(all(not image.isVisible() for image in panel.waterfall_images))
        self.window.view_tabs.setCurrentIndex(1)
        self.assertIs(self.window.capture.stream, stream)
        self.assertFalse(self.window.peak_hold_checkbox.isVisible())
        self.assertTrue(self.window.history_seconds.isVisible())
        for channel, image in enumerate(panel.waterfall_images):
            self.assertTrue(image.isVisible())
            self.assertIsNotNone(image.image)
            self.assertFalse(panel.curves[channel].isVisible())
            self.assertFalse(panel.peak_curves[channel].isVisible())
            self.assertEqual(panel.waterfall_color_bars[channel].levels(), (justfan.DB_FLOOR, 0))
            self.assertEqual(panel.plots[channel].getViewBox().viewRange()[1], [-15, 0])
        self.window.view_tabs.setCurrentIndex(0)
        self.assertTrue(self.window.peak_hold_checkbox.isVisible())
        self.assertFalse(self.window.history_seconds.isVisible())
        self.assertIs(self.window.capture.stream, stream)
        for channel, held in enumerate(panel.peak_curves):
            self.assertTrue(held.isVisible())
            self.assertTrue(panel.curves[channel].isVisible())
            self.assertFalse(panel.waterfall_images[channel].isVisible())
            self.assertAlmostEqual(float(held.getData()[1][0]), 20 * np.log10(0.5))

    def test_waterfall_keeps_unrendered_tones_on_separate_channels(self) -> None:
        settings = self.window.capture.settings
        stream = self.window.capture.stream
        assert settings is not None and isinstance(stream, FakeStream)
        phase = 2 * np.pi * np.arange(settings.block_size) / settings.block_size
        first = np.column_stack((0.5 * np.sin(32 * phase), np.zeros(settings.block_size)))
        later = np.column_stack((np.zeros(settings.block_size), 0.25 * np.sin(96 * phase)))
        stream.callback(first, settings.block_size, None, "")
        stream.callback(later, settings.block_size, None, "")
        self.window._update_measurement()
        self.window.view_tabs.setCurrentIndex(1)
        images = [image.image for image in self.window.spectra.waterfall_images]
        assert images[0] is not None and images[1] is not None
        self.assertAlmostEqual(float(images[0][:, 32].max()), 20 * np.log10(0.5 / np.sqrt(2)), places=3)
        self.assertAlmostEqual(float(images[1][:, 96].max()), 20 * np.log10(0.25 / np.sqrt(2)), places=3)
        self.assertLess(float(images[1][:, 32].max()), -90)
        self.assertLess(float(images[0][:, 96].max()), -90)

    def test_waterfall_duration_stop_reset_and_restart(self) -> None:
        self.window.view_tabs.setCurrentIndex(1)
        self.window.history_seconds.setValue(30)
        self.push(0.5)
        for plot in self.window.spectra.plots:
            self.assertEqual(plot.getViewBox().viewRange()[1], [-30, 0])
        self.push(0.25)
        self.window._stop_capture()
        stopped = self.window.spectra.waterfall_images[0].image
        assert stopped is not None
        before = stopped.copy()
        self.window._update_measurement()
        np.testing.assert_array_equal(self.window.spectra.waterfall_images[0].image, before)
        self.window.reset_button.click()
        reset = self.window.spectra.waterfall_images[0].image
        assert reset is not None
        self.assertAlmostEqual(float(reset.max()), 20 * np.log10(0.25), places=4)
        np.testing.assert_array_equal(reset[:-1], justfan.DB_FLOOR)
        self.window.start_capture()
        self.assertEqual(self.window.view_tabs.currentIndex(), 1)
        self.assertEqual(self.window.spectra.waterfall.seconds, 30)
        self.assertIsNone(self.window.spectra.waterfall_images[0].image)
        self.push(0.1)
        restarted = self.window.spectra.waterfall_images[0].image
        assert restarted is not None
        self.assertAlmostEqual(float(restarted.max()), -20, places=4)

    def test_waterfall_shares_frequency_range_and_channel_visibility(self) -> None:
        self.window.view_tabs.setCurrentIndex(1)
        self.window.frequency_minimum.setValue(1000)
        self.window.frequency_maximum.setValue(8000)
        self.window.channel_widgets[0].enabled_checkbox.setChecked(False)
        self.push(0.5)
        panel = self.window.spectra
        self.assertEqual([plot.isVisible() for plot in panel.plots], [False, True])
        for plot in panel.plots:
            self.assertEqual(plot.getViewBox().viewRange()[0], [1000, 8000])
        self.window.channel_widgets[0].enabled_checkbox.setChecked(True)
        image = panel.waterfall_images[0].image
        assert image is not None
        self.assertAlmostEqual(float(image.max()), 20 * np.log10(0.5), places=4)

    def test_waterfall_compact_layout_scrolls_without_clipping_plot_axes(self) -> None:
        assert self.application is not None
        self.query.return_value = [justfan.InputDevice(3, "Four-channel microphone", "Test API", 4, 48000, 0.01)]
        self.window.refresh_devices()
        self.window.view_tabs.setCurrentIndex(1)
        self.window.resize(860, 560)
        self.window.splitter.setSizes([350, 503])
        self.application.processEvents()
        panel = self.window.spectra
        self.assertGreaterEqual(panel.content.height(), panel.grid.minimumSize().height())
        self.assertTrue(panel.verticalScrollBar().isVisible())
        self.assertFalse(panel.horizontalScrollBar().isVisible())
        for index, plot in enumerate(panel.plots):
            bounds = justfan.QRect(plot.x(), plot.y(), plot.width(), plot.height())
            self.assertTrue(panel.content.rect().contains(bounds))
            for other in panel.plots[index + 1:]:
                self.assertFalse(plot.geometry().intersects(other.geometry()))

    def test_waterfall_view_changes_do_not_replay_previous_device_audio(self) -> None:
        self.push(0.5)
        self.window.device_combo.setCurrentIndex(self.window.device_combo.findText(self.devices[1].label, Qt.MatchFlag.MatchEndsWith))
        self.window.view_tabs.setCurrentIndex(1)
        self.window.history_seconds.setValue(30)
        self.assertEqual(self.window._last_frames, -1)
        self.assertEqual(len(self.window.spectra.plots), 1)
        self.assertIsNone(self.window.spectra.waterfall_images[0].image)
        self.query.return_value = []
        self.window.refresh_devices()
        self.window.view_tabs.setCurrentIndex(0)
        self.window.view_tabs.setCurrentIndex(1)
        self.window.history_seconds.setValue(60)
        self.assertEqual(self.window.spectra.plots, [])

    def test_reset_zoom_restores_all_spectra_without_resetting_measurement(self) -> None:
        self.assertIsInstance(self.window.reset_zoom_button, type(self.window.reset_button))
        self.assertEqual(self.window.reset_zoom_button.text(), "Reset zoom")
        self.assertFalse(self.window.reset_zoom_button.icon().isNull())
        self.push(0.5)
        self.push(0.25)
        self.window.frequency_minimum.setValue(1000)
        self.window.frequency_maximum.setValue(8000)
        self.window.channel_widgets[1].enabled_checkbox.setChecked(False)
        panel = self.window.spectra
        stream = self.window.capture.stream
        buffer = self.window.capture.buffer
        assert buffer is not None and panel.waterfall.data is not None
        before = buffer.snapshot()
        assert before is not None
        history = panel.waterfall.data.copy()
        for plot in panel.plots:
            view = plot.getViewBox()
            for factor in (0.5, 1.5, 0.6):
                view.scaleBy((factor, factor))
            view.setRange(xRange=(2000, 4000), yRange=(-60, -20), padding=0)
        self.window.reset_zoom_button.click()
        for plot in panel.plots:
            self.assertEqual(plot.getViewBox().viewRange(), [[1000, 8000], [justfan.DB_FLOOR - 5, 0]])
        self.assertEqual([plot.isHidden() for plot in panel.plots], [False, True])
        self.assertEqual((self.window.frequency_minimum.value(), self.window.frequency_maximum.value()), (1000, 8000))
        self.assertIs(self.window.capture.stream, stream)
        after = buffer.snapshot()
        assert after is not None
        self.assertEqual(after.frames, before.frames)
        np.testing.assert_array_equal(after.peaks, before.peaks)
        np.testing.assert_array_equal(after.spectrum_peaks, before.spectrum_peaks)
        np.testing.assert_array_equal(panel.waterfall.data, history)

    def test_reset_zoom_restores_waterfall_and_cached_spectrum_scale_while_stopped(self) -> None:
        panel = self.window.spectra
        for plot in panel.plots:
            plot.getViewBox().setRange(yRange=(-90, -15), padding=0)
        self.window.view_tabs.setCurrentIndex(1)
        self.window.history_seconds.setValue(30)
        self.window.frequency_minimum.setValue(500)
        self.window.frequency_maximum.setValue(9000)
        self.push(0.5)
        self.window._stop_capture()
        assert panel.waterfall.data is not None
        history = panel.waterfall.data.copy()
        for plot in panel.plots:
            plot.getViewBox().setRange(xRange=(2000, 3500), yRange=(-8, -2), padding=0)
        self.window.reset_zoom_button.click()
        for plot in panel.plots:
            self.assertEqual(plot.getViewBox().viewRange(), [[500, 9000], [-30, 0]])
        np.testing.assert_array_equal(panel.waterfall.data, history)
        self.assertIsNone(self.window.capture.stream)
        self.window.view_tabs.setCurrentIndex(0)
        for plot in panel.plots:
            self.assertEqual(plot.getViewBox().viewRange(), [[500, 9000], [justfan.DB_FLOOR - 5, 0]])

    def test_reset_zoom_caps_frequency_range_at_nyquist(self) -> None:
        panel = justfan.SpectrumPanel()
        self.addCleanup(panel.close)
        panel.set_frequency_range(1000, 10_000)
        panel.set_channels(1, 16000)
        panel.plots[0].getViewBox().setRange(xRange=(2000, 4000), yRange=(-80, -20), padding=0)
        panel.reset_zoom()
        self.assertEqual(panel.plots[0].getViewBox().viewRange(), [[1000, 8000], [justfan.DB_FLOOR - 5, 0]])

    def test_reset_zoom_is_disabled_without_input_devices(self) -> None:
        self.assertTrue(self.window.reset_zoom_button.isEnabled())
        self.query.return_value = []
        self.window.refresh_devices()
        self.assertFalse(self.window.reset_zoom_button.isEnabled())
        self.window.spectra.reset_zoom()

    def test_spectrum_defaults_to_frequency_range_up_to_10_khz(self) -> None:
        self.assertEqual(self.window.spectra.plots[0].getViewBox().viewRange()[0], [0, 10_000])

    def test_static_spectrum_axes_use_device_cache(self) -> None:
        axis_cache_kib = 7680 * 4320 * 4 * 2 // 1024
        self.assertGreaterEqual(QPixmapCache.cacheLimit(), axis_cache_kib * 2)
        for plot in self.window.spectra.plots:
            for name in ("left", "bottom"):
                self.assertEqual(plot.getAxis(name).cacheMode(), QGraphicsItem.CacheMode.DeviceCoordinateCache)

    def test_axis_cache_refreshes_after_range_and_size_changes(self) -> None:
        assert self.application is not None
        self.push(0.5)
        self.push(0.25)
        for width, height, minimum, maximum in (
            (1360, 820, 0, 10_000),
            (1100, 680, 1000, 8000),
            (1700, 1000, 500, 16_000),
        ):
            with self.subTest(width=width, minimum=minimum, maximum=maximum):
                self.window.resize(width, height)
                self.window.frequency_minimum.setValue(minimum)
                self.window.frequency_maximum.setValue(maximum)
                for plot in self.window.spectra.plots:
                    plot.setYRange(-110, -5, padding=0)
                self.application.processEvents()
                for plot in self.window.spectra.plots:
                    cached = plot.viewport().grab().toImage()
                    self.assertEqual(plot.getAxis("bottom").range, [minimum, maximum])
                    self.assertEqual(plot.getAxis("left").range, [-110, -5])
                    axes = [plot.getAxis(name) for name in ("left", "bottom")]
                    for axis in axes:
                        axis.setCacheMode(QGraphicsItem.CacheMode.NoCache)
                    for axis in axes:
                        axis.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
                    rebuilt = plot.viewport().grab().toImage()
                    self.assertEqual(cached, rebuilt)

    def test_spectrum_default_frequency_range_is_capped_at_nyquist(self) -> None:
        panel = justfan.SpectrumPanel()
        self.addCleanup(panel.close)
        panel.set_channels(1, 16000)
        self.assertEqual(panel.plots[0].getViewBox().viewRange()[0], [0, 8000])

    def test_frequency_range_controls_update_all_spectra(self) -> None:
        self.window.frequency_minimum.setValue(1000)
        self.window.frequency_maximum.setValue(8000)
        self.assertEqual(self.window.spectra.frequency_range, (1000, 8000))
        for plot in self.window.spectra.plots:
            self.assertEqual(plot.getViewBox().viewRange()[0], [1000, 8000])

    def test_frequency_range_controls_are_capped_by_device_nyquist(self) -> None:
        self.window.frequency_maximum.setValue(30_000)
        self.window.device_combo.setCurrentIndex(
            self.window.device_combo.findText(self.devices[1].label, Qt.MatchFlag.MatchEndsWith)
        )
        self.assertEqual(self.window.frequency_maximum.value(), 22_050)
        self.assertEqual(self.window.spectra.plots[0].getViewBox().viewRange()[0][1], 22_050)

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

    def test_fps_is_preserved_on_stop_and_reset_on_restart(self) -> None:
        with patch.object(justfan, "perf_counter", return_value=100.0) as clock:
            self.window.start_capture()
            clock.return_value = 101.0
            self.push()
            self.assertEqual(self.window.fps_label.text(), "FPS 1.0")
            settings = self.window.capture.settings
            assert settings is not None
            for _ in range((int(settings.sample_rate) + settings.block_size - 1) // settings.block_size - 1):
                self.push()
            elapsed = self.window.elapsed_label.text()
            self.assertEqual(elapsed, "Measured 00:00:01")
            self.window._toggle_capture()
            self.assertEqual(self.window.fps_label.text(), "FPS 1.0")
            self.assertEqual(self.window.elapsed_label.text(), elapsed)
            clock.return_value = 102.0
            self.window._toggle_capture()
            self.assertEqual(self.window.fps_label.text(), "FPS 0.0")
            clock.return_value = 103.0
            self.push()
            self.assertEqual(self.window.fps_label.text(), "FPS 1.0")
            self.query.return_value = []
            self.window.refresh_devices()
            self.assertEqual(self.window.fps_label.text(), "FPS 1.0")

    def test_compact_controls_remove_duplicate_title_and_reuse_parameter_row(self) -> None:
        controls = self.window.findChild(QWidget, "controls")
        assert controls is not None
        control_layout = controls.layout()
        assert control_layout is not None
        self.assertEqual(self.window.windowTitle(), f"JustFan v{justfan.VERSION}")
        self.assertFalse(any(label.text() == "JustFan" for label in controls.findChildren(QLabel)))
        self.assertEqual(control_layout.count(), 2)
        parameter_item = control_layout.itemAt(1)
        device_item = control_layout.itemAt(0)
        assert parameter_item is not None and device_item is not None
        parameters = parameter_item.layout()
        assert isinstance(parameters, justfan.QGridLayout)
        device_row = device_item.layout()
        assert device_row is not None
        self.assertEqual(device_row.indexOf(self.window.commands_button), 3)
        option_item = parameters.itemAtPosition(0, 4)
        action_item = parameters.itemAtPosition(1, 4)
        assert option_item is not None and action_item is not None
        options = option_item.layout()
        actions = action_item.layout()
        assert options is not None and actions is not None
        option_widgets = [item.widget() for index in range(options.count()) if (item := options.itemAt(index)) is not None]
        action_widgets = [item.widget() for index in range(actions.count()) if (item := actions.itemAt(index)) is not None]
        self.assertEqual(option_widgets, [self.window.record_checkbox, None, self.window.reset_button])
        self.assertEqual(action_widgets, [self.window.start_button, self.window.apply_button])
        assert self.application is not None
        for width in (1360, 860):
            with self.subTest(width=width):
                self.window.resize(width, 680)
                self.application.processEvents()
                self.assertGreaterEqual(self.window.width(), controls.minimumSizeHint().width())
                widgets = (
                    self.window.sample_rate, self.window.channel_count, self.window.fft_size,
                    self.window.latency, self.window.commands_button, self.window.apply_button,
                    self.window.start_button, self.window.record_checkbox, self.window.reset_button,
                )
                for index, widget in enumerate(widgets):
                    self.assertTrue(controls.rect().contains(widget.geometry()))
                    for other in widgets[index + 1:]:
                        self.assertFalse(widget.geometry().intersects(other.geometry()))
                self.assertLess(self.window.commands_button.geometry().bottom(), self.window.record_checkbox.y())
                self.assertLess(self.window.reset_button.geometry().bottom(), self.window.apply_button.y())
                self.assertGreaterEqual(self.window.commands_button.width(), self.window.commands_button.sizeHint().width())

    def test_current_peak_placeholders_and_graphs(self) -> None:
        self.push(0.5)
        self.push(0.25)
        for widget in self.window.channel_widgets:
            self.assertEqual(widget.current_values[0].text(), "-12.0")
            self.assertEqual(widget.peak_values[0].text(), "-6.0")
            assert widget.level_meter.current is not None and widget.level_meter.peak is not None
            self.assertAlmostEqual(widget.level_meter.current, 20 * np.log10(0.25), places=5)
            self.assertAlmostEqual(widget.level_meter.peak, 20 * np.log10(0.5), places=5)
            self.assertTrue(widget.current_values[0].alignment() & Qt.AlignmentFlag.AlignLeft)
            self.assertTrue(widget.peak_values[0].alignment() & Qt.AlignmentFlag.AlignRight)
            for label in widget.current_values[1:] + widget.peak_values[1:]:
                self.assertEqual(label.text(), "--")
        settings = self.window.capture.settings
        assert settings is not None
        frequencies = self.window.spectra.curves[0].getData()[0]
        self.assertGreater(len(frequencies), 0)
        self.assertLessEqual(frequencies[-1], 10_000 + settings.sample_rate / settings.block_size)

    def test_noise_meter_peak_holds_and_resets_while_stopped(self) -> None:
        self.push(0.5)
        self.push(0.25)
        meter = self.window.channel_widgets[0].level_meter
        assert meter.current is not None and meter.peak is not None
        self.assertGreater(meter.peak, meter.current)
        current = meter.current
        self.window._toggle_capture()
        self.assertEqual(meter.current, current)
        self.assertGreater(meter.peak, current)
        self.window._reset_measurement()
        self.assertEqual(meter.current, current)
        self.assertEqual(meter.peak, current)
        self.assertEqual(self.window.channel_widgets[0].peak_values[0].text(), "-12.0")

    def test_channel_checkbox_hides_spectrum_without_losing_readings(self) -> None:
        assert self.application is not None
        panel = self.window.spectra
        stream = self.window.capture.stream
        curves = list(panel.curves)
        checkbox = self.window.channel_widgets[0].enabled_checkbox
        self.assertTrue(all(widget.enabled_checkbox.isChecked() for widget in self.window.channel_widgets))
        self.push(0.5)
        checkbox.click()
        self.application.processEvents()
        self.assertFalse(checkbox.isChecked())
        self.assertTrue(checkbox.isVisible())
        self.assertTrue(checkbox.isEnabled())
        self.assertEqual([plot.isVisible() for plot in panel.plots], [False, True])
        self.assertEqual(panel.grid.count(), 1)
        self.assertEqual(panel.grid.indexOf(panel.plots[0]), -1)
        self.assertIs(panel.grid.itemAtPosition(0, 0).widget(), panel.plots[1])
        self.push(0.75)
        self.push(0.25)
        self.assertEqual(self.window.channel_widgets[0].current_values[0].text(), "-12.0")
        self.assertEqual(self.window.channel_widgets[0].peak_values[0].text(), "-2.5")
        checkbox.click()
        self.application.processEvents()
        self.assertTrue(all(plot.isVisible() for plot in panel.plots))
        self.assertEqual(panel.grid.count(), 2)
        self.assertEqual(panel.curves, curves)
        self.assertIs(self.window.capture.stream, stream)
        for current, held in zip(panel.curves, panel.peak_curves):
            self.assertAlmostEqual(float(current.getData()[1][0]), 20 * np.log10(0.25))
            self.assertAlmostEqual(float(held.getData()[1][0]), 20 * np.log10(0.75))

    def test_all_channels_can_be_disabled_and_reenabled(self) -> None:
        assert self.application is not None
        panel = self.window.spectra
        for widget in self.window.channel_widgets:
            widget.enabled_checkbox.click()
        self.application.processEvents()
        self.assertFalse(any(plot.isVisible() for plot in panel.plots))
        self.assertEqual(panel.grid.count(), 1)
        self.assertEqual(panel.grid.itemAt(0).widget().text(), "No channels enabled")
        self.push()
        self.window.channel_widgets[1].enabled_checkbox.click()
        self.application.processEvents()
        self.assertIsNone(panel._empty)
        self.assertEqual(panel.grid.count(), 1)
        self.assertIs(panel.grid.itemAtPosition(0, 0).widget(), panel.plots[1])
        self.assertEqual([plot.isVisible() for plot in panel.plots], [False, True])
        self.window.channel_widgets[0].enabled_checkbox.click()
        self.application.processEvents()
        self.assertTrue(all(plot.isVisible() for plot in panel.plots))
        self.assertEqual(panel.grid.count(), 2)

    def test_channel_filter_reflows_in_channel_order_across_resizes(self) -> None:
        assert self.application is not None
        self.query.return_value = [justfan.InputDevice(3, "Eight-channel microphone", "Test API", 8, 48000, 0.01)]
        self.window.refresh_devices()
        panel = self.window.spectra
        enabled_channels = (1, 4, 7)
        for channel, widget in enumerate(self.window.channel_widgets):
            widget.enabled_checkbox.setChecked(channel in enabled_channels)
        for width, sizes in ((1360, [350, 1000]), (860, [500, 360]), (1500, [350, 1150])):
            with self.subTest(width=width):
                self.window.resize(width, 680)
                self.window.splitter.setSizes(sizes)
                self.application.processEvents()
                self.assertEqual(panel.grid.count(), len(enabled_channels))
                self.assertEqual([plot.isVisible() for plot in panel.plots], [channel in enabled_channels for channel in range(8)])
                for index, channel in enumerate(enabled_channels):
                    self.assertEqual(panel.grid.getItemPosition(panel.grid.indexOf(panel.plots[channel]))[:2], divmod(index, panel.columns))
                self.assertFalse(panel.horizontalScrollBar().isVisible())

    def test_channel_filter_cooperates_with_spectrum_focus(self) -> None:
        assert self.application is not None
        panel = self.window.spectra
        plot = panel.plots[1]
        self.window.channel_widgets[0].enabled_checkbox.click()
        for focused in (True, False):
            QTest.mouseDClick(plot.viewport(), Qt.MouseButton.LeftButton, pos=plot.viewport().rect().center())
            self.application.processEvents()
            self.assertIs(panel._focused_plot, plot if focused else None)
            self.assertEqual([item.isVisible() for item in panel.plots], [False, True])
            self.assertEqual(panel.grid.count(), 1)
        self.window.channel_widgets[0].enabled_checkbox.click()
        QTest.mouseDClick(plot.viewport(), Qt.MouseButton.LeftButton, pos=plot.viewport().rect().center())
        self.application.processEvents()
        self.window.channel_widgets[1].enabled_checkbox.click()
        self.application.processEvents()
        self.assertIsNone(panel._focused_plot)
        self.assertEqual([item.isVisible() for item in panel.plots], [True, False])
        self.assertEqual(panel.grid.count(), 1)
        self.assertIs(panel.grid.itemAtPosition(0, 0).widget(), panel.plots[0])

    def test_channel_selection_survives_restart_and_is_device_specific(self) -> None:
        self.window.channel_widgets[1].enabled_checkbox.click()
        self.window.start_capture()
        self.assertEqual([widget.enabled_checkbox.isChecked() for widget in self.window.channel_widgets], [True, False])
        self.assertEqual([plot.isHidden() for plot in self.window.spectra.plots], [False, True])
        self.window.device_combo.setCurrentIndex(self.window.device_combo.findText(self.devices[1].label, Qt.MatchFlag.MatchEndsWith))
        self.assertTrue(self.window.channel_widgets[0].enabled_checkbox.isChecked())
        self.window.channel_widgets[0].enabled_checkbox.click()
        self.window.device_combo.setCurrentIndex(self.window.device_combo.findText(self.devices[0].label, Qt.MatchFlag.MatchEndsWith))
        self.assertEqual([widget.enabled_checkbox.isChecked() for widget in self.window.channel_widgets], [True, False])
        self.window.channel_count.setValue(1)
        self.window.start_capture()
        self.assertTrue(self.window.channel_widgets[0].enabled_checkbox.isChecked())
        self.window.channel_count.setValue(2)
        self.window.start_capture()
        self.assertFalse(self.window.channel_widgets[1].enabled_checkbox.isChecked())
        self.assertTrue(self.window.spectra.plots[1].isHidden())

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
            for curve, expected in (
                (self.window.spectra.curves[channel], current[:, channel]),
                (self.window.spectra.peak_curves[channel], held[:, channel]),
            ):
                frequencies, levels = curve.getData()
                indices = np.searchsorted(analyzer.frequencies, frequencies)
                np.testing.assert_allclose(levels, expected[indices])

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
        self.window.start_capture()
        settings = self.window.capture.settings
        stream = self.window.capture.stream
        analyzer = self.window.analyzer
        assert settings is not None and stream is not None and analyzer is not None
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
                frequencies, levels = held.getData()
                indices = np.searchsorted(analyzer.frequencies, frequencies)
                np.testing.assert_array_equal(levels, frame.spectrum_peaks[indices, channel])
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
        self.check.reset_mock()
        self.window.refresh_devices()
        combo = self.window.device_combo
        self.assertEqual([combo.itemData(row).index for row in range(combo.count())], [6, 9, 3, 11, 12])
        self.assertEqual(
            [combo.itemText(row) for row in range(combo.count())],
            [f"{row + 1}. {combo.itemData(row).label}" for row in range(combo.count())],
        )
        self.assertEqual(combo.currentData().index, 3)
        self.assertEqual(combo.currentText(), f"3. {self.devices[0].label}")
        self.check.assert_not_called()
        combo.setCurrentIndex(combo.findText(devices[0].label, Qt.MatchFlag.MatchEndsWith))
        self.window.refresh_devices()
        self.assertEqual(combo.currentData().index, 12)
        self.assertEqual(combo.currentText(), f"5. {devices[0].label}")
        self.check.assert_not_called()

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

    def test_virtual_device_caps_initial_channel_count(self) -> None:
        self.query.return_value = [justfan.InputDevice(10, "default", "ALSA", 128, 48000, 0.01)]
        self.window.refresh_devices()
        self.assertEqual(self.window.channel_count.maximum(), 128)
        self.assertEqual(self.window.channel_count.value(), justfan.DEFAULT_MAX_CHANNELS)
        self.assertEqual(len(self.window.channel_widgets), justfan.DEFAULT_MAX_CHANNELS)
        self.assertEqual(len(self.window.spectra.plots), justfan.DEFAULT_MAX_CHANNELS)

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

    def test_four_spectra_use_balanced_rows_and_fill_available_height(self) -> None:
        assert self.application is not None
        self.query.return_value = [justfan.InputDevice(3, "Four-channel microphone", "Test API", 4, 48000, 0.01)]
        self.window.refresh_devices()
        self.window.resize(1360, 680)
        self.window.splitter.setSizes([350, 1000])
        self.application.processEvents()
        panel = self.window.spectra
        self.assertEqual(panel.columns, 2)
        self.assertEqual(
            [panel.grid.getItemPosition(panel.grid.indexOf(plot))[:2] for plot in panel.plots],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        )
        self.assertGreater(panel.plots[2].geometry().height(), 180)
        self.assertFalse(panel.verticalScrollBar().isVisible())

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