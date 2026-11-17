"""LUFS-target loudness normalization for MoviePy audio and video clips.

This module implements the integrated loudness meter and the gain stage of
ITU-R BS.1770-4 / EBU R128:

* K-weighting (high shelf and high pass) applied as a single FIR filter,
* 400 ms gating blocks with 75 % overlap,
* an absolute gate at -70 LUFS and a relative gate 10 LU below the ungated
  mean,
* a 4x oversampled true peak meter.

Both meters consume the clip through ``AudioClip.iter_chunks`` and only keep a
fixed-size window in memory, so long clips are processed in blocks and the
measured values do not depend on the chunk size used to read them.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from moviepy.Clip import Clip
from moviepy.decorators import audio_video_effect
from moviepy.Effect import Effect


# ITU-R BS.1770-4 gating constants.
_ABSOLUTE_GATE_LUFS = -70.0
_RELATIVE_GATE_LU = -10.0
_LOUDNESS_OFFSET = -0.691
_DEFAULT_BLOCK_SECONDS = 0.4
_DEFAULT_OVERLAP = 0.75

# The K-weighting high shelf is centred at 1682 Hz, so it requires a sample
# rate well above 2 * 1682 Hz to be represented.
_MIN_SAMPLE_RATE = 8000

# Extra attenuation applied when the true peak ceiling is the limiting factor.
# The BS.1770 4x prototype under-reads the continuous-time peak of material
# close to Nyquist by up to ~0.5 dB, and other compliant meters may read a few
# tenths of a dB higher; this margin keeps exported files under the ceiling
# when checked with a different true peak implementation.
_PEAK_SAFETY_DB = 0.5

# 4x oversampling polyphase filter from ITU-R BS.1770-4, Annex 2 (48 taps).
_TRUE_PEAK_PHASES = (
    np.array(
        [
            0.0017089843750,
            0.0109863281250,
            -0.0196533203125,
            0.0332031250000,
            -0.0594482421875,
            0.1373291015625,
            0.9721679687500,
            -0.1022949218750,
            0.0476074218750,
            -0.0266113281250,
            0.0148925781250,
            -0.0083007812500,
        ]
    ),
    np.array(
        [
            -0.0291748046875,
            0.0292968750000,
            -0.0517578125000,
            0.0891113281250,
            -0.1665039062500,
            0.4650878906250,
            0.7797851562500,
            -0.2003173828125,
            0.1015625000000,
            -0.0582275390625,
            0.0330810546875,
            -0.0189208984375,
        ]
    ),
    np.array(
        [
            -0.0189208984375,
            0.0330810546875,
            -0.0582275390625,
            0.1015625000000,
            -0.2003173828125,
            0.7797851562500,
            0.4650878906250,
            -0.1665039062500,
            0.0891113281250,
            -0.0517578125000,
            0.0292968750000,
            -0.0291748046875,
        ]
    ),
    np.array(
        [
            -0.0083007812500,
            0.0148925781250,
            -0.0266113281250,
            0.0476074218750,
            -0.1022949218750,
            0.9721679687500,
            0.1373291015625,
            -0.0594482421875,
            0.0332031250000,
            -0.0196533203125,
            0.0109863281250,
            0.0017089843750,
        ]
    ),
)

# Filtering and analysis are done on fixed-size internal blocks. Fixing these
# sizes makes every result a pure function of the input samples, never of the
# chunk size used by the caller.
_FILTER_BLOCK = 65536

# Length of the truncated K-weighting impulse response, in seconds. The slowest
# pole (the 38 Hz high pass) decays by over 20 time constants in this span.
_KERNEL_SECONDS = 0.1

# Polarity weights of the BS.1770 channel gains (mono, stereo, then surround).
_CHANNEL_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.41, 1.41])


def _biquad_coefficients(filter_type, sample_rate):
    """Return the normalized ``(b, a)`` coefficients of a K-weighting filter.

    The coefficients are the analytic ("DeMan") forms of the analogue
    prototypes in ITU-R BS.1770-4, valid at any supported sample rate.
    """
    if filter_type == "high_shelf":
        gain_db = 3.99984385397
        q = 0.7071752369554193
        frequency = 1681.9744509555319
        k = np.tan(np.pi * frequency / sample_rate)
        vh = 10.0 ** (gain_db / 20.0)
        vb = vh**0.499666774155
        denominator = 1.0 + k / q + k * k
        b = np.array(
            [
                (vh + vb * k / q + k * k) / denominator,
                2.0 * (k * k - vh) / denominator,
                (vh - vb * k / q + k * k) / denominator,
            ]
        )
        a = np.array(
            [
                1.0,
                2.0 * (k * k - 1.0) / denominator,
                (1.0 - k / q + k * k) / denominator,
            ]
        )
        return b, a

    if filter_type == "high_pass":
        q = 0.5003270373253953
        frequency = 38.13547087613982
        k = np.tan(np.pi * frequency / sample_rate)
        denominator = 1.0 + k / q + k * k
        b = np.array([1.0, -2.0, 1.0])
        a = np.array(
            [
                1.0,
                2.0 * (k * k - 1.0) / denominator,
                (1.0 - k / q + k * k) / denominator,
            ]
        )
        return b, a

    raise ValueError(f"Unknown filter type: {filter_type!r}")


def _biquad_impulse_response(b, a, length):
    """Impulse response of a normalized biquad, computed with its recurrence."""
    response = np.zeros(length)
    x1 = x2 = y1 = y2 = 0.0
    for index in range(length):
        x0 = 1.0 if index == 0 else 0.0
        y0 = b[0] * x0 + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        response[index] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
    return response


@lru_cache(maxsize=32)
def _k_weighting_kernel(sample_rate):
    """Truncated FIR impulse response of the two cascaded K-weighting filters.

    Expressing the IIR pair as one finite impulse response lets the whole clip
    be filtered with vectorized FFT convolution while producing the same output
    as filtering the complete signal in one pass.
    """
    length = max(256, int(math.ceil(sample_rate * _KERNEL_SECONDS)))
    shelf_b, shelf_a = _biquad_coefficients("high_shelf", sample_rate)
    high_pass_b, high_pass_a = _biquad_coefficients("high_pass", sample_rate)
    shelf = _biquad_impulse_response(shelf_b, shelf_a, length)
    high_pass = _biquad_impulse_response(high_pass_b, high_pass_a, length)
    kernel = np.convolve(shelf, high_pass)[:length]
    kernel.flags.writeable = False
    return kernel


def _next_power_of_two(value):
    return 1 << (int(value) - 1).bit_length()


def _fft_convolution_valid(signal, kernel, spectrum_cache):
    """Causal convolution of ``signal`` and ``kernel`` with ``len(signal)`` out."""
    fft_size = _next_power_of_two(len(signal) + len(kernel) - 1)
    spectrum = spectrum_cache.get(fft_size)
    if spectrum is None:
        spectrum = np.fft.rfft(kernel, fft_size)
        spectrum_cache[fft_size] = spectrum
    convolved = np.fft.irfft(np.fft.rfft(signal, fft_size) * spectrum, fft_size)
    start = len(kernel) - 1
    return convolved[start : len(signal)]


def _as_frames(samples):
    """Return a ``(samples, channels)`` float64 view of an audio frame batch."""
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2:
        raise ValueError(
            "AudioLoudnessNormalize expects mono or multichannel audio frames, "
            f"got an array with shape {array.shape}."
        )
    return array


class _StreamingConvolver:
    """Streaming FIR filter with exact overlap between consecutive blocks."""

    def __init__(self, kernel):
        self.kernel = kernel
        self._spectra = {}
        self._channels = None
        self._history = None
        self._pending = None

    def _ensure_channels(self, channels):
        if self._channels is None:
            self._channels = channels
            self._history = np.zeros((len(self.kernel) - 1, channels))
            self._pending = np.zeros((0, channels))
        elif channels != self._channels:
            raise ValueError(
                "AudioLoudnessNormalize requires a constant channel count, got "
                f"{channels} channels after {self._channels}."
            )

    def _filter_block(self, block):
        extension = np.vstack([self._history, block])
        output = np.empty(block.shape, dtype=np.float64)
        for channel in range(block.shape[1]):
            output[:, channel] = _fft_convolution_valid(
                extension[:, channel], self.kernel, self._spectra
            )
        self._history = extension[-(len(self.kernel) - 1) :].copy()
        return output

    def feed(self, frames):
        self._ensure_channels(frames.shape[1])
        self._pending = np.concatenate([self._pending, frames])
        blocks = []
        while self._pending.shape[0] >= _FILTER_BLOCK:
            blocks.append(self._filter_block(self._pending[:_FILTER_BLOCK]))
            self._pending = self._pending[_FILTER_BLOCK:]
        if blocks:
            return np.concatenate(blocks, axis=0)
        return np.zeros((0, self._channels))

    def flush(self):
        if self._pending is None or not self._pending.shape[0]:
            return np.zeros((0, self._channels or 1))
        pending = self._pending
        self._pending = np.zeros((0, pending.shape[1]))
        return self._filter_block(pending)


class _TruePeakMeter:
    """Streaming sample peak and 4x oversampled true peak meter.

    The four polyphase branches of the BS.1770 interpolator are applied to
    every sample, so each block only needs the last ``taps - 1`` samples of
    the previous block as history. The result therefore does not depend on how
    the caller chunks the signal.
    """

    def __init__(self):
        self.true_peak = 0.0
        self.sample_peak = 0.0
        self._taps = max(len(phases) for phases in _TRUE_PEAK_PHASES)
        self._history = None

    def feed(self, frames):
        if self._history is None:
            self._history = np.zeros((self._taps - 1, frames.shape[1]))
        extension = np.vstack([self._history, frames])
        sample_peak = float(np.max(np.abs(frames)))
        if sample_peak > self.sample_peak:
            self.sample_peak = sample_peak
        for phases in _TRUE_PEAK_PHASES:
            for channel in range(frames.shape[1]):
                interpolated = np.convolve(extension[:, channel], phases, mode="valid")
                if interpolated.size:
                    peak = float(np.max(np.abs(interpolated)))
                    if peak > self.true_peak:
                        self.true_peak = peak
        self._history = extension[-(self._taps - 1) :].copy()


def _amplitude_to_db(amplitude):
    if not amplitude > 0:
        return -math.inf
    return 20.0 * math.log10(amplitude)


class _LoudnessAnalyzer:
    """Streaming integrated loudness and true peak analyzer."""

    def __init__(self, sample_rate, block_seconds, overlap):
        self.sample_rate = sample_rate
        self.block_seconds = block_seconds
        self.step = 1.0 - overlap
        self._convolver = _StreamingConvolver(_k_weighting_kernel(sample_rate))
        self._true_peak = _TruePeakMeter()
        self._buffer = None
        self._buffer_start = 0
        self._total = 0
        self._block_index = 0
        self._block_energies = []
        self._channels = None

    def _block_bounds(self, index):
        start = int(self.block_seconds * self.step * index * self.sample_rate)
        end = int(self.block_seconds * (self.step * index + 1) * self.sample_rate)
        return start, end

    def _append_filtered(self, filtered):
        if filtered.shape[0] == 0:
            return
        if self._buffer is None:
            self._buffer = filtered
        else:
            self._buffer = np.concatenate([self._buffer, filtered])
        self._total += filtered.shape[0]
        while True:
            start, end = self._block_bounds(self._block_index)
            if end > self._total:
                break
            offset = start - self._buffer_start
            block = self._buffer[offset : offset + (end - start)]
            self._block_energies.append(np.mean(block * block, axis=0))
            self._block_index += 1
            next_start, _ = self._block_bounds(self._block_index)
            trim = next_start - self._buffer_start
            if trim > 0:
                self._buffer = self._buffer[trim:]
                self._buffer_start = next_start

    def feed(self, samples, first_sample_index=0):
        frames = _as_frames(samples)
        if frames.shape[0] == 0:
            return
        if self._channels is None:
            if frames.shape[1] > len(_CHANNEL_WEIGHTS):
                raise ValueError(
                    "AudioLoudnessNormalize supports at most "
                    f"{len(_CHANNEL_WEIGHTS)} channels, got {frames.shape[1]}."
                )
            self._channels = frames.shape[1]
        elif frames.shape[1] != self._channels:
            raise ValueError(
                "AudioLoudnessNormalize requires a constant channel count, got "
                f"{frames.shape[1]} channels after {self._channels}."
            )
        finite = np.isfinite(frames)
        if not finite.all():
            flat = int(np.argmax(~finite))
            sample, channel = divmod(flat, frames.shape[1])
            raise ValueError(
                "AudioLoudnessNormalize cannot measure non-finite samples "
                f"(first at sample {first_sample_index + sample}, "
                f"channel {channel})."
            )
        self._true_peak.feed(frames)
        self._append_filtered(self._convolver.feed(frames))

    def finish(self):
        self._append_filtered(self._convolver.flush())
        channels = self._channels or 1
        if self._total == 0:
            return _Measurement(
                integrated_lufs=-math.inf,
                true_peak_db=-math.inf,
                sample_peak_db=-math.inf,
                block_count=0,
                sample_count=0,
                channels=channels,
            )

        if self._block_index:
            energies = np.array(self._block_energies)
        else:
            # Clips shorter than one gating block are measured as a single
            # block covering the whole clip. This is intentionally defined
            # behaviour instead of an error: very short clips are common in
            # a playlist and can still be normalized.
            energies = np.mean(self._buffer * self._buffer, axis=0)[None, :]

        weights = _CHANNEL_WEIGHTS[: energies.shape[1]]
        with np.errstate(divide="ignore", invalid="ignore"):
            block_loudness = _LOUDNESS_OFFSET + 10.0 * np.log10(energies @ weights)

        above_absolute = block_loudness >= _ABSOLUTE_GATE_LUFS
        if not np.any(above_absolute):
            integrated = -math.inf
        else:
            gated_mean = np.mean(energies[above_absolute], axis=0)
            relative_threshold = (
                _LOUDNESS_OFFSET + 10.0 * np.log10(np.sum(gated_mean * weights))
            ) + _RELATIVE_GATE_LU
            above_relative = (block_loudness > relative_threshold) & (
                block_loudness > _ABSOLUTE_GATE_LUFS
            )
            final_mean = np.mean(energies[above_relative], axis=0)
            integrated = _LOUDNESS_OFFSET + 10.0 * np.log10(
                np.sum(final_mean * weights)
            )

        return _Measurement(
            integrated_lufs=float(integrated),
            true_peak_db=_amplitude_to_db(self._true_peak.true_peak),
            sample_peak_db=_amplitude_to_db(self._true_peak.sample_peak),
            block_count=self._block_index,
            sample_count=self._total,
            channels=channels,
        )


class _Measurement:
    """Internal measurement container used before the public dataclass."""

    __slots__ = (
        "integrated_lufs",
        "true_peak_db",
        "sample_peak_db",
        "block_count",
        "sample_count",
        "channels",
    )

    def __init__(
        self,
        integrated_lufs,
        true_peak_db,
        sample_peak_db,
        block_count,
        sample_count,
        channels,
    ):
        self.integrated_lufs = integrated_lufs
        self.true_peak_db = true_peak_db
        self.sample_peak_db = sample_peak_db
        self.block_count = block_count
        self.sample_count = sample_count
        self.channels = channels


@dataclass(frozen=True)
class LoudnessMeasurement:
    """Loudness and peak measurements of a clip.

    Attributes
    ----------

    integrated_lufs
      Gated integrated loudness in LUFS. ``-inf`` when the clip is silent or
      contains no sample.

    true_peak_db
      True peak (4x oversampled) in dBTP. ``-inf`` for digital silence.

    sample_peak_db
      Highest absolute sample value in dBFS. ``-inf`` for digital silence.

    block_count
      Number of 400 ms gating blocks used by the BS.1770 algorithm. ``0``
      means the clip was shorter than one block and was measured as a single
      block.

    sample_count
      Number of audio samples analyzed.

    duration
      Analyzed duration in seconds.

    channels
      Number of audio channels.
    """

    integrated_lufs: float
    true_peak_db: float
    sample_peak_db: float
    block_count: int
    sample_count: int
    duration: float
    channels: int

    @property
    def is_silent(self):
        """Whether the integrated loudness is undefined because of silence."""
        return not math.isfinite(self.integrated_lufs)


def _resolve_audio(clip):
    """Return the audio clip to measure, resolving video clips."""
    if hasattr(clip, "audio"):
        if clip.audio is None:
            raise ValueError(
                "AudioLoudnessNormalize requires a clip with an audio track."
            )
        return clip.audio
    return clip


def _validate_chunk_size(chunk_size):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")


def _validate_analysis_parameters(chunk_size, block_seconds, overlap):
    _validate_chunk_size(chunk_size)
    if (
        isinstance(block_seconds, bool)
        or not isinstance(block_seconds, (int, float, np.integer, np.floating))
        or not math.isfinite(block_seconds)
        or block_seconds <= 0
    ):
        raise ValueError("block_seconds must be a positive finite number.")
    if (
        isinstance(overlap, bool)
        or not isinstance(overlap, (int, float, np.integer, np.floating))
        or not math.isfinite(overlap)
        or not 0.0 <= overlap < 1.0
    ):
        raise ValueError("overlap must be a finite number in [0, 1).")


def measure_loudness(
    clip,
    chunk_size=50000,
    block_seconds=_DEFAULT_BLOCK_SECONDS,
    overlap=_DEFAULT_OVERLAP,
    logger=None,
):
    """Measure integrated loudness and true peak of an audio or video clip.

    The clip is read through ``AudioClip.iter_chunks`` in chunks of at most
    ``chunk_size`` samples, so the complete audio is never held in memory.
    Gating blocks are aligned on the absolute sample timeline, which makes the
    result independent of the chunk size and stable across block boundaries.

    Parameters
    ----------

    clip
      An ``AudioClip``, or a video clip whose ``audio`` track is measured.

    chunk_size
      Maximum number of samples requested from the clip at a time.

    block_seconds
      Gating block length in seconds. BS.1770 uses 400 ms.

    overlap
      Overlap between consecutive gating blocks. BS.1770 uses 0.75.

    logger
      Progress bar logger forwarded to ``AudioClip.iter_chunks``.

    Returns
    -------

    LoudnessMeasurement
      Integrated loudness, true peak, sample peak and analyzed duration.

    Raises
    ------

    ValueError
      If the clip has no duration, no positive sample rate, different channel
      counts between chunks, non-finite samples, or an unsupported sample rate.
    """
    audio = _resolve_audio(clip)
    _validate_analysis_parameters(chunk_size, block_seconds, overlap)

    if audio.duration is None:
        raise ValueError("AudioLoudnessNormalize requires a clip with a duration.")
    fps = getattr(audio, "fps", None)
    if fps is None or isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
        raise ValueError(
            "AudioLoudnessNormalize requires a clip with a positive sample rate."
        )
    sample_rate = int(round(fps))
    if sample_rate < _MIN_SAMPLE_RATE:
        raise ValueError(
            "AudioLoudnessNormalize requires a sample rate of at least "
            f"{_MIN_SAMPLE_RATE} Hz, got {fps}."
        )

    total_samples = int(fps * audio.duration)
    if total_samples <= 0:
        return LoudnessMeasurement(
            integrated_lufs=-math.inf,
            true_peak_db=-math.inf,
            sample_peak_db=-math.inf,
            block_count=0,
            sample_count=0,
            duration=0.0,
            channels=getattr(audio, "nchannels", 1),
        )

    analyzer = _LoudnessAnalyzer(sample_rate, block_seconds, overlap)
    index = 0
    for chunk in audio.iter_chunks(chunksize=chunk_size, logger=logger):
        analyzer.feed(chunk, first_sample_index=index)
        index += len(chunk)
    measurement = analyzer.finish()
    return LoudnessMeasurement(
        integrated_lufs=measurement.integrated_lufs,
        true_peak_db=measurement.true_peak_db,
        sample_peak_db=measurement.sample_peak_db,
        block_count=measurement.block_count,
        sample_count=measurement.sample_count,
        duration=measurement.sample_count / float(sample_rate),
        channels=measurement.channels,
    )


def _check_finite_number(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


@dataclass
class AudioLoudnessNormalize(Effect):
    """Normalize a clip's loudness to a target LUFS level.

    The integrated loudness of the whole clip is measured following
    ITU-R BS.1770-4 / EBU R128, then one constant gain brings it to
    ``target_lufs``. When that gain would push the output true peak above
    ``true_peak_db``, the gain is reduced instead, so the output never clips
    and its loudness stays below the target. The effect can be applied to an
    ``AudioClip`` or to a video clip, whose frames and duration are kept
    untouched while only the audio is normalized.

    Parameters
    ----------

    target_lufs
      Desired integrated loudness in LUFS. Must be finite and lie in
      ``(-70, 0]``. Defaults to -23 LUFS, the EBU R128 broadcast target.

    true_peak_db
      Maximum output true peak in dBTP. Must be finite and at most 0.
      Defaults to -1 dBTP.

    chunk_size
      Maximum number of samples read at a time while measuring. Results do
      not depend on this value.

    logger
      Progress bar logger used during measurement, e.g. ``"bar"``.

    Notes
    -----

    Edge cases have defined behaviour:

    * silent clips (integrated loudness undefined) are returned unchanged;
    * clips shorter than one 400 ms gating block are measured as a single
      block and are still normalized;
    * clips with no sample, no duration or a sample rate below 8 kHz raise
      ``ValueError``;
    * non-finite samples (NaN or infinity) raise ``ValueError``;
    * a target that cannot be reached without exceeding ``true_peak_db`` is
      peak-limited: the true peak is respected and the achieved loudness is
      lower than the target;
    * an invalid ``target_lufs``, ``true_peak_db`` or ``chunk_size`` raises
      ``ValueError``.

    Examples
    --------

    >>> from moviepy import *
    >>> clip = AudioFileClip("podcast.mp3")
    >>> clip = clip.with_effects(
    ...     [afx.AudioLoudnessNormalize(target_lufs=-16, true_peak_db=-1)]
    ... )
    >>> clip.write_audiofile("podcast_normalized.wav")

    Video clips keep their frames and duration:

    >>> video = VideoFileClip("movie.mp4").with_effects(
    ...     [afx.AudioLoudnessNormalize(target_lufs=-23)]
    ... )
    """

    target_lufs: float = -23.0
    true_peak_db: float = -1.0
    chunk_size: int = 50000
    logger: str = None

    def _validate(self):
        target = _check_finite_number(self.target_lufs, "target_lufs")
        if not -70.0 < target <= 0.0:
            raise ValueError("target_lufs must be in (-70, 0].")
        peak = _check_finite_number(self.true_peak_db, "true_peak_db")
        if peak > 0.0:
            raise ValueError("true_peak_db must be at most 0.")
        _validate_chunk_size(self.chunk_size)

    @audio_video_effect
    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        self._validate()
        measurement = measure_loudness(
            clip,
            chunk_size=self.chunk_size,
            logger=self.logger,
        )
        if measurement.is_silent:
            # No measurable loudness: there is nothing to normalize.
            return clip

        gain_db = self.target_lufs - measurement.integrated_lufs
        if math.isfinite(measurement.true_peak_db):
            # Leave a small margin so a differently implemented true peak
            # meter also reads the exported file below the ceiling.
            peak_limited_db = (
                self.true_peak_db - measurement.true_peak_db - _PEAK_SAFETY_DB
            )
            gain_db = min(gain_db, peak_limited_db)

        gain = 10.0 ** (gain_db / 20.0)
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError(
                "AudioLoudnessNormalize computed a non-finite or non-positive "
                f"gain ({gain!r}) for the supplied target and peak ceiling."
            )
        if abs(gain - 1.0) < 1e-12:
            return clip
        return clip.transform(
            lambda get_frame, t: gain * get_frame(t),
            keep_duration=True,
        )
