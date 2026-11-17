from dataclasses import dataclass

import numpy as np

from moviepy.Clip import Clip
from moviepy.decorators import audio_video_effect
from moviepy.Effect import Effect
from moviepy.tools import convert_to_seconds


@dataclass
class AudioDuck(Effect):
    """Automatically lowers ("ducks") the volume of a music clip while a
    voice-over clip is playing, then smoothly restores it during the pauses.

    The voice-over is analyzed over short sliding windows. Every window whose
    RMS energy is above ``threshold`` is considered as speech. While speech is
    detected the music gain is smoothly lowered towards ``target_gain`` (the
    attenuation depth), and smoothly brought back to ``1`` when the
    voice-over falls silent again. The transitions are driven by an
    attack/release one-pole smoother, so a narration made of two separated
    segments produces two smooth ducking bands with a full recovery in
    between.

    Parameters
    ----------

    voiceover : AudioClip
      The clip carrying the voice-over that drives the ducking. It may be
      mono or stereo, may start later than the music through its ``start``
      attribute, and can use a different ``fps`` than the music clip. Only the
      part of the voice-over overlapping the music is analyzed.

    target_gain : float, optional
      Linear gain applied to the music while the voice-over is active,
      between ``0`` and ``1``. ``0.2`` means the music drops to 20% of its
      original volume (about -14 dB). Default is ``0.2``.

    threshold : float, optional
      RMS energy above which a voice-over window is considered as speech.
      Must be strictly positive. A higher threshold makes the effect less
      sensitive to quiet background noise. Default is ``0.05``.

    attack : float, optional
      Time in seconds for the gain to move most of the way towards
      ``target_gain`` once speech is detected. ``0`` gives an instantaneous
      (sample-accurate) ducking. Cannot be negative. Default is ``0.2``.

    release : float, optional
      Time in seconds for the gain to move most of the way back to ``1`` once
      the voice-over becomes silent. ``0`` gives an instantaneous recovery.
      Cannot be negative. Default is ``0.5``.

    analysis_window : float, optional
      Length in seconds of the sliding windows used to measure the
      voice-over energy. Cannot be negative. Default is ``0.05``.

    Examples
    --------

    .. code:: python

        from moviepy import AudioFileClip, afx, CompositeAudioClip

        music = AudioFileClip("music.mp3")
        narration = AudioFileClip("narration.mp3").with_start(2)

        # Lower the music to 15% while the narrator speaks.
        ducked_music = music.with_effects(
            [afx.AudioDuck(narration, target_gain=0.15)]
        )
        final_audio = CompositeAudioClip([ducked_music, narration])
    """

    voiceover: Clip
    target_gain: float = 0.2
    threshold: float = 0.05
    attack: float = 0.2
    release: float = 0.5
    analysis_window: float = 0.05

    def __post_init__(self):
        self.attack = convert_to_seconds(self.attack)
        self.release = convert_to_seconds(self.release)
        self.analysis_window = convert_to_seconds(self.analysis_window)

        if not isinstance(self.target_gain, (int, float)) or not (
            0 <= self.target_gain <= 1
        ):
            raise ValueError("'target_gain' must be a number between 0 and 1")

        if not isinstance(self.threshold, (int, float)) or self.threshold <= 0:
            raise ValueError("'threshold' must be a strictly positive number")

        if not isinstance(self.attack, (int, float)) or self.attack < 0:
            raise ValueError("'attack' must be a non-negative number of seconds")

        if not isinstance(self.release, (int, float)) or self.release < 0:
            raise ValueError("'release' must be a non-negative number of seconds")

        if (
            not isinstance(self.analysis_window, (int, float))
            or self.analysis_window < 0
        ):
            raise ValueError(
                "'analysis_window' must be a non-negative number of seconds"
            )

    def _build_envelope(self, clip):
        """Compute the per-sample gain envelope applied to the music."""
        fps = clip.fps
        n_samples = int(clip.duration * fps)
        times = np.arange(n_samples) / fps

        # Sample the voice-over on the music timeline, using zeros outside of
        # the voice-over time range so different start times, durations and
        # sampling rates all work transparently.
        voice_start = self.voiceover.start or 0
        voice_end = (
            voice_start + self.voiceover.duration
            if self.voiceover.duration is not None
            else None
        )
        in_range = times >= voice_start
        if voice_end is not None:
            in_range &= times < voice_end

        voice_samples = np.zeros(n_samples)
        if in_range.any():
            voice_frames = np.asarray(
                self.voiceover.to_soundarray(times[in_range] - voice_start, fps=fps),
                dtype=float,
            )
            # RMS across channels: stereo narration drives a single envelope.
            # Mono clips return a 1-D array and must be kept element-wise.
            if voice_frames.ndim > 1:
                rms = np.sqrt(np.mean(np.square(voice_frames), axis=-1))
            else:
                rms = np.abs(voice_frames)
            voice_samples[in_range] = rms

        # Short-window RMS energy through a normalized causal moving average:
        # the window only looks at past and current samples, so the energy of
        # a speech segment cannot leak onto the samples preceding it.
        window_size = max(1, int(round(self.analysis_window * fps)))
        window = np.ones(window_size) / window_size
        energy = np.sqrt(
            np.convolve(np.square(voice_samples), window, mode="full")[:n_samples]
        )

        targets = np.where(energy > self.threshold, self.target_gain, 1.0)

        # A completely silent voice-over never ducks the music: return the
        # clip unchanged instead of building a pointless all-ones envelope.
        if not np.any(targets < 1.0):
            return None

        # Attack/release one-pole smoothing. ``attack``/``release`` are the
        # time constants of the filter, so a zero time constant means the
        # gain snaps instantly to its target.
        def smoothing_factor(time_constant):
            if time_constant <= 0:
                return 1.0
            return 1.0 - np.exp(-1.0 / (time_constant * fps))

        attack_factor = smoothing_factor(self.attack)
        release_factor = smoothing_factor(self.release)

        # Process each run of constant target as a whole with the closed form
        # of the one-pole filter, which keeps the computation vectorized.
        envelope = np.empty(n_samples)
        run_starts = np.concatenate(([0], np.flatnonzero(np.diff(targets)) + 1))
        run_ends = np.append(run_starts[1:], n_samples)
        # If the narration is already speaking when the music starts, the
        # music must begin already ducked instead of replaying the attack.
        previous_gain = targets[0]
        for start, run_end in zip(run_starts, run_ends):
            target = targets[start]
            factor = attack_factor if target < previous_gain else release_factor
            run_length = run_end - start
            envelope[start:run_end] = target + (previous_gain - target) * (
                1.0 - factor
            ) ** np.arange(1, run_length + 1)
            previous_gain = envelope[run_end - 1]

        return envelope

    @audio_video_effect
    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        if clip.duration is None:
            raise ValueError("Attribute 'duration' not set on the music clip")
        if clip.duration <= 0:
            raise ValueError("The music clip duration must be strictly positive")
        if self.voiceover.duration is None:
            raise ValueError("Attribute 'duration' not set on the voice-over clip")
        if self.voiceover.duration <= 0:
            raise ValueError("The voice-over clip duration must be strictly positive")
        if not hasattr(clip, "fps") or clip.fps is None:
            raise ValueError("Attribute 'fps' not set on the music clip")

        # A unity target gain means the music is left untouched.
        if self.target_gain >= 1:
            return clip

        envelope = self._build_envelope(clip)
        if envelope is None:
            return clip

        def ducked_filter(get_frame, t):
            frame = get_frame(t)
            if np.isscalar(t):
                index = min(max(int(clip.fps * t), 0), len(envelope) - 1)
                return frame * envelope[index]

            indices = np.clip(
                np.round(clip.fps * np.asarray(t)).astype(int),
                0,
                len(envelope) - 1,
            )
            gain = envelope[indices]
            if np.ndim(frame) > 1:
                gain = gain.reshape(-1, 1)
            return frame * gain

        return clip.transform(ducked_filter, keep_duration=True)
