from dataclasses import dataclass

import numpy as np

from moviepy.Clip import Clip
from moviepy.decorators import audio_video_effect
from moviepy.Effect import Effect


class _SidechainEnvelope:
    """Streaming, order-independent sidechain gain envelope.

    The envelope is a pure function of the trigger clip's samples and the
    target clip's frame rate. To avoid keeping the whole trigger in memory and
    to make random frame access give exactly the same result as sequential
    rendering, the envelope state is stored at fixed-size block boundaries and
    replayed from the closest checkpoint whenever an earlier sample is needed.

    Parameters
    ----------

    trigger
      The AudioClip driving the ducking (e.g. the voice-over track).

    fps
      Frame rate at which the envelope is sampled, i.e. the target clip fps.

    threshold
      Absolute trigger level above which the gain is pushed down.

    target_gain
      Linear gain (between 0 and 1) the envelope tends towards while the
      trigger is above the threshold.

    attack_time, release_time
      Time constants, in seconds, of the first-order (exponential) attack and
      release filters. A value of 0 gives an instantaneous response.

    block_size
      Number of envelope samples between two checkpoints and the maximum number
      of trigger samples requested in one batch.
    """

    def __init__(
        self,
        trigger,
        fps,
        threshold,
        target_gain,
        attack_time,
        release_time,
        block_size,
    ):
        self.trigger = trigger
        self.fps = fps
        self.threshold = threshold
        self.target_gain = target_gain
        self.block_size = max(1, int(block_size))

        # First-order filter coefficients: alpha = exp(-dt / tau), so after
        # ``tau`` seconds the envelope has covered ~63% of the remaining gap.
        self.attack_alpha = (
            0.0 if attack_time <= 0 else float(np.exp(-1.0 / (fps * attack_time)))
        )
        self.release_alpha = (
            0.0 if release_time <= 0 else float(np.exp(-1.0 / (fps * release_time)))
        )

        # Checkpoints: canonical sample index -> (gain, mode).
        # Block 0 starts at sample 0 with an untouched gain and no trigger.
        self._checkpoints = {0: (1.0, "release")}

    def _trigger_levels(self, indices):
        """Return the per-sample trigger level (max across channels).

        Times are mapped on the composition clock so that clips with different
        start/end times stay aligned; times outside the trigger are silent.
        Trigger samples are always requested in batches no larger than
        ``block_size``, the whole trigger is never read at once.
        """
        times = indices / self.fps
        local_times = times - self.trigger.start

        start = self.trigger.start
        end = self.trigger.end
        if end is None and self.trigger.duration is not None:
            end = start + self.trigger.duration
        if end is None:
            active = np.ones(len(local_times), dtype=bool)
        else:
            active = (times >= start) & (times < end)
        active &= local_times >= 0

        levels = np.zeros(len(local_times))
        if not active.any():
            return levels

        requested = local_times[active]
        frames = np.asarray(self.trigger.get_frame(requested), dtype=float)
        if frames.ndim == 1:
            levels[active] = np.abs(frames)
        else:
            levels[active] = np.abs(frames).max(axis=1)
        return levels

    def gain_at(self, indices):
        """Gain envelope values at the given canonical sample indices."""
        indices = np.asarray(indices)
        scalar = indices.ndim == 0
        indices = np.atleast_1d(indices).astype(np.int64)

        flat = indices.ravel()
        if not flat.size:
            return np.empty(0)
        first = int(flat.min())
        last = int(flat.max())

        # Replay from the most advanced checkpoint that is not past the first
        # requested sample, so every query is always reached by moving forward.
        checkpoint_index = max(k for k in self._checkpoints if k <= first)
        gain, mode = self._checkpoints[checkpoint_index]

        gains = np.empty(last + 1 - checkpoint_index)
        cursor = checkpoint_index
        out = 0
        while cursor <= last:
            if cursor > checkpoint_index and cursor % self.block_size == 0:
                self._checkpoints[cursor] = (gain, mode)

            block_end = min(last + 1, cursor + self.block_size)
            block_indices = np.arange(cursor, block_end)
            for level in self._trigger_levels(block_indices):
                if level > self.threshold:
                    mode = "attack"
                else:
                    mode = "release"

                target = self.target_gain if mode == "attack" else 1.0
                alpha = self.attack_alpha if mode == "attack" else self.release_alpha
                # Convex combination: the gain stays within [target_gain, 1]
                # and is continuous when crossing the threshold, so silences
                # and short pauses never produce jumps.
                gain = alpha * gain + (1.0 - alpha) * target
                gains[out] = gain
                out += 1
            cursor = block_end

        envelope = gains[flat - checkpoint_index]
        if scalar:
            return float(envelope[0])
        return envelope.reshape(indices.shape)


@dataclass
class SidechainGain(Effect):
    """Ducks (lowers) the clip volume driven by another audio track.

    While the ``trigger`` clip is louder than ``threshold``, the gain applied
    to the target clip smoothly moves towards ``target_gain``; when the trigger
    becomes silent again, the gain smoothly recovers to 1. This is the
    sidechain compression / "ducking" effect used to lower background music
    under a voice-over.

    Unlike :class:`MultiplyVolume`, which applies a fixed factor, the applied
    gain is an envelope computed from another track. The effect returns a
    composable, exportable :class:`AudioClip` and never loads the trigger track
    entirely in memory: the trigger is read in small batches, and rendering the
    clip frame by frame in any order gives exactly the same result as a
    sequential export.

    Parameters
    ----------

    trigger : AudioClip
      The audio track driving the ducking (for instance a narration clip).
      Its own start/end times and sample rate are taken into account.

    threshold : float, optional
      Absolute trigger level above which the target gain starts being reduced,
      in the ``[0, 1]`` range. Defaults to ``0.02``.

    target_gain : float, optional
      Linear gain the envelope tends towards while the trigger is above the
      threshold. Must be in the ``[0, 1]`` range; ``0`` mutes the target while
      the trigger is loud, ``1`` disables ducking. Defaults to ``0.2``.

    attack_time : float, optional
      Time constant in seconds for the gain to move towards ``target_gain``
      once the trigger exceeds the threshold. ``0`` gives an instantaneous
      attack. Defaults to ``0.05``.

    release_time : float, optional
      Time constant in seconds for the gain to recover to 1 once the trigger
      falls below the threshold. ``0`` gives an instantaneous release.
      Defaults to ``0.5``.

    block_size : int, optional
      Number of envelope samples per streaming block / checkpoint. The trigger
      is never read in batches larger than this. Lower values reduce memory
      usage, higher values reduce the number of trigger reads. Defaults to
      ``50000``.

    Examples
    --------

    .. code:: python

        from moviepy import AudioFileClip, afx

        music = AudioFileClip("background_music.ogg")
        voiceover = AudioFileClip("narration.ogg").with_start(2)

        # Lower the music to 20% while the narration is louder than 2%
        ducked_music = music.with_effects([
            afx.SidechainGain(
                voiceover,
                threshold=0.02,
                target_gain=0.2,
                attack_time=0.05,
                release_time=0.5,
            )
        ])
    """

    trigger: Clip
    threshold: float = 0.02
    target_gain: float = 0.2
    attack_time: float = 0.05
    release_time: float = 0.5
    block_size: int = 50000

    def __post_init__(self):
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not 0.0 <= self.target_gain <= 1.0:
            raise ValueError("target_gain must be between 0 and 1")
        if self.attack_time < 0 or self.release_time < 0:
            raise ValueError("attack_time and release_time must be non-negative")
        if self.block_size < 1:
            raise ValueError("block_size must be a positive integer")
        if not hasattr(self.trigger, "get_frame"):
            raise TypeError("trigger must be an AudioClip")

    @audio_video_effect
    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        if not getattr(clip, "fps", None):
            raise ValueError(
                "SidechainGain requires the target clip to have a frame rate "
                "(fps) set so the envelope can be sampled."
            )
        fps = clip.fps
        envelope = _SidechainEnvelope(
            trigger=self.trigger,
            fps=fps,
            threshold=self.threshold,
            target_gain=self.target_gain,
            attack_time=self.attack_time,
            release_time=self.release_time,
            block_size=self.block_size,
        )

        def sidechain(get_frame, t):
            frame = get_frame(t)
            times = np.atleast_1d(np.asarray(t, dtype=float))
            # The envelope is a function of a canonical sample index only, so
            # any time maps to one and only one gain, whatever the query order.
            indices = np.maximum(0, np.floor(fps * times)).astype(np.int64)
            gains = envelope.gain_at(indices)

            if np.ndim(t) == 0:
                return frame * float(gains[0])
            if np.ndim(frame) == 1:  # mono batch, frame shape (n_samples,)
                return frame * gains
            return frame * gains[:, np.newaxis]  # stereo batch (n_samples, nch)

        return clip.transform(sidechain, keep_duration=True)
