"""Tests for the LUFS-target loudness normalization effect."""

import math
import os
import wave

import numpy as np

import pytest

from moviepy import (
    AudioArrayClip,
    AudioClip,
    AudioFileClip,
    VideoFileClip,
    concatenate_audioclips,
)
from moviepy.audio.fx import AudioNormalize
from moviepy.audio.fx.AudioLoudnessNormalize import (
    _TRUE_PEAK_PHASES,
    AudioLoudnessNormalize,
    measure_loudness,
)


RATE = 44100


def _rbj_k_weighting(rate):
    """K-weighting coefficients from the RBJ audio EQ cookbook.

    Deliberately independent from the implementation under test, which uses
    the analytic ITU ("DeMan") coefficients. Both sets agree within a few
    hundredths of a dB across the audio band.
    """
    filters = []

    # High shelf: +4 dB above 1500 Hz.
    gain = 4.0
    q = 1.0 / np.sqrt(2.0)
    frequency = 1500.0
    a = 10.0 ** (gain / 40.0)
    w0 = 2.0 * np.pi * frequency / rate
    alpha = np.sin(w0) / (2.0 * q)
    b = np.array(
        [
            a * ((a + 1) + (a - 1) * np.cos(w0) + 2 * np.sqrt(a) * alpha),
            -2 * a * ((a - 1) + (a + 1) * np.cos(w0)),
            a * ((a + 1) + (a - 1) * np.cos(w0) - 2 * np.sqrt(a) * alpha),
        ]
    )
    den = np.array(
        [
            (a + 1) - (a - 1) * np.cos(w0) + 2 * np.sqrt(a) * alpha,
            2 * ((a - 1) - (a + 1) * np.cos(w0)),
            (a + 1) - (a - 1) * np.cos(w0) - 2 * np.sqrt(a) * alpha,
        ]
    )
    filters.append((b / den[0], den / den[0]))

    # High pass at 38 Hz.
    q = 0.5
    frequency = 38.0
    w0 = 2.0 * np.pi * frequency / rate
    alpha = np.sin(w0) / (2.0 * q)
    b = np.array([(1 + np.cos(w0)) / 2, -(1 + np.cos(w0)), (1 + np.cos(w0)) / 2])
    den = np.array([1 + alpha, -2 * np.cos(w0), 1 - alpha])
    filters.append((b / den[0], den / den[0]))
    return filters


def _reference_integrated_loudness(samples, rate):
    """Independent BS.1770-4 integrated loudness implemented in the DFT."""
    data = np.asarray(samples, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    length = data.shape[0]
    frequencies = np.fft.rfftfreq(length, 1.0 / rate)
    z = np.exp(-2j * np.pi * frequencies / rate)
    filtered = np.empty_like(data)
    for channel in range(data.shape[1]):
        spectrum = np.fft.rfft(data[:, channel])
        for b, a in _rbj_k_weighting(rate):
            response = (b[0] + b[1] * z + b[2] * z * z) / (
                a[0] + a[1] * z + a[2] * z * z
            )
            spectrum = spectrum * response
        filtered[:, channel] = np.fft.irfft(spectrum, length)

    block = int(round(0.4 * rate))
    step = int(round(0.1 * rate))
    energies = np.array(
        [
            np.mean(filtered[s : s + block] ** 2, axis=0)
            for s in range(0, length - block + 1, step)
        ]
    )
    loudness = -0.691 + 10.0 * np.log10(energies.sum(axis=1))
    above_absolute = loudness >= -70.0
    gated = np.mean(energies[above_absolute], axis=0)
    relative = -0.691 + 10.0 * np.log10(gated.sum()) - 10.0
    kept = (loudness > relative) & (loudness > -70.0)
    return -0.691 + 10.0 * np.log10(np.mean(energies[kept], axis=0).sum())


def _reference_true_peak(samples, oversampling=16):
    """True peak by band-limited oversampling using a windowed sinc kernel."""
    data = np.asarray(samples, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    taps = 32
    n = np.arange(taps * oversampling) - taps * oversampling // 2
    kernel = np.sinc(n / oversampling) * np.kaiser(len(n), 10.0)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    padded = np.concatenate(
        [np.zeros((taps, data.shape[1])), data, np.zeros((taps, data.shape[1]))]
    )
    for phase in range(oversampling):
        coefficients = kernel[phase::oversampling]
        coefficients = coefficients / coefficients.sum()
        for channel in range(data.shape[1]):
            interpolated = np.convolve(padded[:, channel], coefficients, "valid")
            if interpolated.size:
                peak = max(peak, float(np.max(np.abs(interpolated))))
    return peak


def _stereo_tone(frequency=440.0, amplitude=0.2, duration=4.0, rate=RATE):
    t = np.arange(int(rate * duration)) / rate
    left = amplitude * np.sin(2 * np.pi * frequency * t)
    right = 0.6 * amplitude * np.sin(2 * np.pi * (frequency * 1.5) * t + 0.4)
    return np.stack([left, right], axis=1)


def _write_wav(path, samples, rate=RATE):
    data = np.asarray(samples)
    if data.ndim == 1:
        data = data[:, None]
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(os.fspath(path), "wb") as handle:
        handle.setnchannels(data.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def _read_wav(path):
    with wave.open(os.fspath(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    assert width == 2
    data = np.frombuffer(frames, dtype="<i2").astype(float) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels)
    return data, rate


def test_measure_matches_independent_reference():
    samples = _stereo_tone()
    clip = AudioArrayClip(samples, fps=RATE)
    measurement = measure_loudness(clip)

    reference = _reference_integrated_loudness(samples, RATE)
    assert measurement.integrated_lufs == pytest.approx(reference, abs=0.2)
    assert measurement.channels == 2
    assert measurement.duration == pytest.approx(4.0)


def test_measure_mono_sine_absolute_level():
    # A 1 kHz sine of amplitude 0.4 measures about -11 LUFS after K-weighting.
    frame_function = lambda t: 0.4 * np.sin(2 * np.pi * 1000 * np.asarray(t))
    clip = AudioClip(frame_function, duration=3.0, fps=RATE)
    assert clip.nchannels == 1

    measurement = measure_loudness(clip)
    assert measurement.channels == 1
    assert measurement.integrated_lufs == pytest.approx(-10.96, abs=0.15)
    assert measurement.true_peak_db == pytest.approx(20 * math.log10(0.4), abs=0.1)


def test_measure_is_stable_across_chunk_sizes():
    clip = AudioArrayClip(_stereo_tone(duration=6.0), fps=RATE)
    measurements = [
        measure_loudness(clip, chunk_size=chunk_size)
        for chunk_size in (1, 997, 5000, 50000, 1000000)
    ]
    first = measurements[0]
    for measurement in measurements[1:]:
        assert measurement.integrated_lufs == pytest.approx(
            first.integrated_lufs, abs=1e-9
        )
        assert measurement.true_peak_db == pytest.approx(first.true_peak_db, abs=1e-9)
        assert measurement.sample_peak_db == pytest.approx(
            first.sample_peak_db, abs=1e-9
        )
        assert measurement.block_count == first.block_count


def test_normalize_reaches_target():
    samples = _stereo_tone(amplitude=0.05, duration=5.0)
    clip = AudioArrayClip(samples, fps=RATE)
    normalized = clip.with_effects(
        [AudioLoudnessNormalize(target_lufs=-20.0, true_peak_db=-1.0)]
    )

    measurement = measure_loudness(normalized)
    assert measurement.integrated_lufs == pytest.approx(-20.0, abs=0.1)
    assert measurement.duration == pytest.approx(clip.duration)
    assert normalized.duration == pytest.approx(clip.duration)

    exported = normalized.to_soundarray()
    assert exported.shape == samples.shape
    reference = _reference_integrated_loudness(exported, RATE)
    assert reference == pytest.approx(-20.0, abs=0.3)


def test_normalize_mono_clip():
    frame_function = lambda t: 0.05 * np.sin(2 * np.pi * 330 * np.asarray(t))
    clip = AudioClip(frame_function, duration=4.0, fps=RATE)
    normalized = clip.with_effects([AudioLoudnessNormalize(target_lufs=-18.0)])

    assert normalized.nchannels == 1
    assert normalized.duration == pytest.approx(4.0)
    assert measure_loudness(normalized).integrated_lufs == pytest.approx(-18.0, abs=0.1)


def test_normalize_respects_true_peak_ceiling():
    # A quiet bed with one loud sample: reaching -10 LUFS would need a gain
    # that pushes the true peak over -1 dBTP, so the gain must be limited.
    t = np.arange(int(RATE * 4)) / RATE
    bed = 0.02 * np.sin(2 * np.pi * 440 * t)
    bed[int(2 * RATE)] = 1.0
    samples = np.stack([bed, bed], axis=1)
    clip = AudioArrayClip(samples, fps=RATE)
    normalized = clip.with_effects(
        [AudioLoudnessNormalize(target_lufs=-10.0, true_peak_db=-1.0)]
    )

    exported = normalized.to_soundarray()
    peak_db = 20 * math.log10(_reference_true_peak(exported))
    assert peak_db <= -1.0
    # The limiter engaged: the requested loudness was not reached.
    assert measure_loudness(normalized).integrated_lufs < -10.0
    # But it got close to the ceiling instead of over-attenuating.
    assert peak_db > -2.5


def test_normalize_true_peak_uses_intersample_peaks():
    # A quarter-sample-rate sine with a phase offset: the sampled values stay
    # below the analog amplitude while the interpolated waveform reaches it.
    t = np.arange(int(RATE * 3)) / RATE
    amplitude = 0.7
    samples = np.stack([amplitude * np.sin(2 * np.pi * RATE / 4 * t + 0.3)] * 2, axis=1)
    clip = AudioArrayClip(samples, fps=RATE)
    measurement = measure_loudness(clip)
    assert measurement.sample_peak_db == pytest.approx(
        20 * math.log10(amplitude * np.cos(0.3)), abs=0.1
    )
    assert measurement.true_peak_db == pytest.approx(
        20 * math.log10(amplitude), abs=0.2
    )
    assert measurement.true_peak_db > measurement.sample_peak_db


def test_normalize_preserves_video_frames_and_duration(util):
    with VideoFileClip("media/chaplin.mp4") as video:
        before = measure_loudness(video.audio)
        normalized = video.with_effects([AudioLoudnessNormalize(target_lufs=-20.0)])
        assert normalized.duration == pytest.approx(video.duration)
        assert normalized.fps == video.fps
        assert normalized.size == video.size
        for timestamp in (0.2, 4.1):
            np.testing.assert_array_equal(
                video.get_frame(timestamp), normalized.get_frame(timestamp)
            )
        after = measure_loudness(normalized.audio)
        assert after.integrated_lufs == pytest.approx(-20.0, abs=0.1)
        assert after.integrated_lufs != pytest.approx(before.integrated_lufs)


def test_normalize_silence_is_left_unchanged():
    clip = AudioArrayClip(np.zeros((RATE, 2)), fps=RATE)
    measurement = measure_loudness(clip)
    assert measurement.is_silent
    assert measurement.integrated_lufs == -math.inf
    assert measurement.true_peak_db == -math.inf

    normalized = clip.with_effects([AudioLoudnessNormalize()])
    np.testing.assert_array_equal(normalized.to_soundarray(), clip.to_soundarray())


def test_normalize_very_short_clip():
    frame_function = lambda t: 0.2 * np.sin(2 * np.pi * 440 * np.asarray(t))
    clip = AudioClip(frame_function, duration=0.05, fps=RATE)
    measurement = measure_loudness(clip)
    # Shorter than one 400 ms block: measured as a single block, not an error.
    assert measurement.block_count == 0
    assert math.isfinite(measurement.integrated_lufs)

    normalized = clip.with_effects([AudioLoudnessNormalize(target_lufs=-20.0)])
    assert normalized.duration == pytest.approx(clip.duration)
    assert measure_loudness(normalized).integrated_lufs == pytest.approx(-20.0, abs=0.1)


def test_normalize_empty_clip():
    clip = AudioClip(lambda t: np.array([]), duration=0.0, fps=RATE)
    measurement = measure_loudness(clip)
    assert measurement.is_silent
    assert measurement.sample_count == 0


def test_normalize_requires_duration_and_supported_sample_rate():
    with pytest.raises(ValueError, match="duration"):
        measure_loudness(AudioClip(lambda t: np.sin(t), fps=RATE))

    with pytest.raises(ValueError, match="sample rate"):
        measure_loudness(AudioClip(lambda t: np.sin(t), duration=1.0))

    with pytest.raises(ValueError, match="sample rate"):
        measure_loudness(AudioClip(lambda t: np.sin(t), duration=1.0, fps=4000))


def test_normalize_exported_peak_stays_under_ceiling_for_bright_material(util):
    # Material close to Nyquist has a sampled peak below its true peak, which
    # alternating loud samples next to a quiet bed exaggerates.
    t = np.arange(int(RATE * 3)) / RATE
    bed = 0.02 * np.sin(2 * np.pi * 17000 * t)
    bed[int(1.5 * RATE)] = 1.0
    samples = np.stack([bed, bed], axis=1)
    clip = AudioArrayClip(samples, fps=RATE)
    normalized = clip.with_effects(
        [AudioLoudnessNormalize(target_lufs=-6.0, true_peak_db=-1.0)]
    )

    filename = os.path.join(util.TMP_DIR, "loudness_peak_ceiling.wav")
    normalized.write_audiofile(filename, codec="pcm_s16le", logger=None)
    exported, rate = _read_wav(filename)
    assert 20 * math.log10(_reference_true_peak(exported)) <= -1.0


def test_normalize_works_on_concatenated_clips():
    first = AudioArrayClip(_stereo_tone(frequency=440.0, duration=2.0), fps=RATE)
    second = AudioArrayClip(
        _stereo_tone(frequency=880.0, amplitude=0.02, duration=2.0), fps=RATE
    )
    concatenated = concatenate_audioclips([first, second])
    normalized = concatenated.with_effects([AudioLoudnessNormalize(target_lufs=-20.0)])

    assert normalized.duration == pytest.approx(concatenated.duration)
    assert measure_loudness(normalized).integrated_lufs == pytest.approx(-20.0, abs=0.1)


def test_normalize_rejects_non_finite_samples():
    samples = np.array([[0.1, 0.2], [0.3, 0.4], [np.nan, 0.5]])
    clip = AudioArrayClip(samples, fps=RATE)
    with pytest.raises(ValueError, match="non-finite"):
        measure_loudness(clip)

    samples = np.array([[0.1, 0.2], [np.inf, 0.4]])
    with pytest.raises(ValueError, match="non-finite"):
        AudioArrayClip(samples, fps=RATE).with_effects([AudioLoudnessNormalize()])


@pytest.mark.parametrize(
    "target_lufs", [-70.0, -70.1, 0.5, 1.0, math.nan, math.inf, -math.inf, "loud"]
)
def test_normalize_rejects_invalid_target(target_lufs):
    clip = AudioArrayClip(_stereo_tone(duration=1.0), fps=RATE)
    with pytest.raises(ValueError, match="target_lufs"):
        clip.with_effects([AudioLoudnessNormalize(target_lufs=target_lufs)])


@pytest.mark.parametrize(
    "true_peak_db", [0.1, 1.0, math.nan, math.inf, -math.inf, "peak"]
)
def test_normalize_rejects_invalid_true_peak(true_peak_db):
    clip = AudioArrayClip(_stereo_tone(duration=1.0), fps=RATE)
    with pytest.raises(ValueError, match="true_peak_db"):
        clip.with_effects([AudioLoudnessNormalize(true_peak_db=true_peak_db)])


@pytest.mark.parametrize("chunk_size", [0, -1, 2.5, "many", None])
def test_measure_rejects_invalid_chunk_size(chunk_size):
    clip = AudioArrayClip(_stereo_tone(duration=1.0), fps=RATE)
    with pytest.raises(ValueError, match="chunk_size"):
        measure_loudness(clip, chunk_size=chunk_size)


def test_normalize_requires_audio_on_video_clip():
    from moviepy.video.VideoClip import ColorClip

    clip = ColorClip(size=(16, 16), color=(0, 0, 0), duration=1.0)
    with pytest.raises(ValueError, match="audio"):
        measure_loudness(clip)


def test_normalize_output_exports_with_expected_duration(util):
    samples = _stereo_tone(amplitude=0.15, duration=3.25)
    clip = AudioArrayClip(samples, fps=RATE)
    normalized = clip.with_effects(
        [AudioLoudnessNormalize(target_lufs=-20.0, true_peak_db=-1.0)]
    )
    filename = os.path.join(util.TMP_DIR, "loudness_normalize.wav")
    normalized.write_audiofile(filename, codec="pcm_s16le", logger=None)

    exported, rate = _read_wav(filename)
    assert rate == RATE
    assert exported.shape[1] == 2
    assert len(exported) / rate == pytest.approx(clip.duration, abs=1.0 / RATE)
    # The exported file measures on target and below the ceiling.
    reference_loudness = _reference_integrated_loudness(exported, rate)
    assert reference_loudness == pytest.approx(-20.0, abs=0.3)
    assert 20 * math.log10(_reference_true_peak(exported)) <= -1.0


def test_audio_normalize_unchanged():
    clip = AudioFileClip("media/crunching.mp3")
    try:
        normalized = clip.with_effects([AudioNormalize()])
        assert normalized.max_volume() == pytest.approx(1.0)
    finally:
        clip.close()


def test_itu_oversampling_phases_are_normalized():
    # The Annex 2 prototype phases must not change the loudness of the signal.
    for phases in _TRUE_PEAK_PHASES:
        assert phases.sum() == pytest.approx(1.0, abs=0.05)
