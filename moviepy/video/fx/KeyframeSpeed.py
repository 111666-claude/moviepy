import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from moviepy.Clip import Clip
from moviepy.Effect import Effect


@dataclass
class KeyframeSpeed(Effect):
    """Retimes a clip following a speed curve defined by keyframes.

    Each keyframe associates a time on the original clip (in seconds) with a
    positive playback speed (``1.0`` is the normal speed). Between two adjacent
    keyframes the speed varies linearly with the *original* time, so the
    source-time-to-output-time mapping (the integral of ``1 / speed``) is a
    smooth, strictly monotonic piecewise logarithmic curve with a continuous
    slope. Before the first keyframe and after the last one, the speed is held
    constant at the speed of the nearest keyframe.

    Unlike :class:`~moviepy.video.fx.MultiplySpeed.MultiplySpeed`, the speed can
    vary over time, which makes it possible to build effects such as a slow
    motion smoothly accelerating back to normal speed. Unlike
    :class:`~moviepy.video.fx.AccelDecel.AccelDecel`, any edit-style speed
    profile can be expressed through arbitrary keyframes.

    The same time mapping is applied to the clip's audio and mask (if any), so
    the picture, the sound and the mask stay synchronized.

    Parameters
    ----------

    keyframes : list of (time, speed) tuples
      Keyframes given as ``(t, speed)`` pairs where ``t`` is a time on the
      original clip in seconds and ``speed`` is the positive playback speed at
      that time (``0.5`` plays twice slower, ``2.0`` twice faster). The pairs
      must be ordered by strictly increasing times. A single keyframe is
      allowed and produces a constant speed, exactly like ``MultiplySpeed``.

    Attributes
    ----------

    output_duration : float or None
      Duration of the retimed clip, computed from the keyframes when the effect
      is applied (``None`` before :meth:`apply` is called).

    Raises
    ------

    ValueError
      If no keyframe is provided, a keyframe is not a ``(time, speed)`` pair,
      the keyframe times are not strictly increasing, a speed is not a positive
      finite number, a keyframe time is not finite, a keyframe lies outside of
      the clip's duration, or the clip has no ``duration`` attribute.

    Examples
    --------

    Play a 4 seconds clip in slow motion (0.25x) for the first second, then
    accelerate linearly back to normal speed between 1s and 3s, and keep normal
    speed until the end::

        myclip.with_effects(
            [vfx.KeyframeSpeed([(0, 0.25), (1, 0.25), (3, 1.0), (4, 1.0)])]
        )
    """

    keyframes: List[Tuple[float, float]]
    output_duration: float = None

    def _validate_keyframes(self, clip_duration):
        """Validate the keyframes and precompute the time mapping tables."""
        if clip_duration is None:
            raise ValueError("Attribute 'duration' not set on the clip")

        if self.keyframes is None or len(self.keyframes) == 0:
            raise ValueError("'keyframes' must contain at least one (time, speed) pair")

        times = []
        speeds = []
        for index, keyframe in enumerate(self.keyframes):
            if (
                not isinstance(keyframe, (tuple, list, np.ndarray))
                or len(keyframe) != 2
            ):
                raise ValueError(
                    "Each keyframe must be a (time, speed) pair, got %r" % (keyframe,)
                )
            t, speed = keyframe

            if (
                not isinstance(t, (int, float, np.integer, np.floating))
                or isinstance(t, bool)
                or not math.isfinite(t)
            ):
                raise ValueError(
                    "Keyframe time at index %d must be a finite number, got %r"
                    % (index, t)
                )
            if (
                not isinstance(speed, (int, float, np.integer, np.floating))
                or isinstance(speed, bool)
                or not math.isfinite(speed)
            ):
                raise ValueError(
                    "Keyframe speed at index %d must be a finite number, got %r"
                    % (index, speed)
                )
            if speed <= 0:
                raise ValueError(
                    "Keyframe speed at index %d must be strictly positive, got %r"
                    % (index, speed)
                )
            if t < 0 or t > clip_duration:
                raise ValueError(
                    "Keyframe time at index %d (%.6f) is outside of the clip's "
                    "duration [0, %.6f]" % (index, t, clip_duration)
                )

            times.append(float(t))
            speeds.append(float(speed))

        for index in range(1, len(times)):
            if times[index] <= times[index - 1]:
                raise ValueError(
                    "Keyframe times must be strictly increasing, got %r"
                    % (self.keyframes,)
                )

        return np.array(times), np.array(speeds)

    def _build_mapping(self, clip_duration):
        """Compute the output time reached at every keyframe.

        The speed is linear in source time on each interval, so the integral of
        ``1 / speed`` giving the output duration is evaluated analytically.
        ``output_times[0]`` holds the output time of the first keyframe, which
        is non-zero when it does not start at the clip's beginning.
        """
        times, speeds = self._validate_keyframes(clip_duration)

        # Constant-speed prefix before the first keyframe.
        output_times = [times[0] / speeds[0]]
        for index in range(1, len(times)):
            t_start, t_end = times[index - 1], times[index]
            v_start, v_end = speeds[index - 1], speeds[index]
            if v_start == v_end:
                segment_duration = (t_end - t_start) / v_start
            else:
                segment_duration = (
                    (t_end - t_start) * np.log(v_end / v_start) / (v_end - v_start)
                )
            output_times.append(output_times[-1] + segment_duration)
        output_times = np.array(output_times)

        # Constant-speed suffix after the last keyframe.
        total_duration = output_times[-1] + (clip_duration - times[-1]) / speeds[-1]

        return times, speeds, output_times, total_duration, clip_duration

    def source_time(self, t):
        """Map an output time ``t`` back to the corresponding source time.

        Works with scalars and numpy arrays of times. This is the function used
        internally to fetch the frames of the retimed clip.
        """
        times, speeds, output_times, total_duration, clip_duration = self._mapping
        scalar_input = np.isscalar(t)
        t = np.asarray(t, dtype=float)
        source = np.empty_like(t)

        # Prefix region: strictly before the first keyframe.
        prefix = t < output_times[0]
        source[prefix] = t[prefix] * speeds[0]

        # Suffix region: strictly after the last keyframe. With a single
        # keyframe the whole clip is constant speed, so there is no suffix.
        suffix = (
            (t > output_times[-1]) if len(times) >= 2 else np.zeros_like(t, dtype=bool)
        )
        source[suffix] = times[-1] + (t[suffix] - output_times[-1]) * speeds[-1]

        # Interior region: keyframe anchors and the intervals between them.
        interior = ~prefix & ~suffix
        if len(times) == 1:
            # A single keyframe defines a constant speed over the whole clip.
            source[interior] = times[0] + (t[interior] - output_times[0]) * speeds[0]
        else:
            for index in range(len(times) - 1):
                in_segment = interior & (t >= output_times[index])
                if index < len(times) - 2:
                    in_segment &= t < output_times[index + 1]
                t0, t1 = times[index], times[index + 1]
                v0, v1 = speeds[index], speeds[index + 1]
                tau0 = output_times[index]
                delta_t = t[in_segment] - tau0
                if v0 == v1:
                    source[in_segment] = t0 + v0 * delta_t
                else:
                    # Inverse of the logarithmic output_time integral:
                    # sigma = t0 + (t1-t0)*v0/(v1-v0) *
                    #         (exp((v1-v0)*delta_t/(t1-t0)) - 1)
                    source[in_segment] = t0 + (
                        (t1 - t0)
                        * v0
                        * (np.exp((v1 - v0) * delta_t / (t1 - t0)) - 1.0)
                        / (v1 - v0)
                    )

        # Guard against floating point drift at the clip boundaries.
        source = np.clip(source, 0.0, clip_duration)
        return float(source) if scalar_input else source

    def output_time(self, source_t):
        """Map a source time ``source_t`` to its time on the retimed clip.

        Works with scalars and numpy arrays of times. It is the exact inverse of
        :meth:`source_time`.
        """
        times, speeds, output_times, total_duration, clip_duration = self._mapping
        scalar_input = np.isscalar(source_t)
        source_t = np.asarray(source_t, dtype=float)
        output = np.empty_like(source_t)

        # Strictly before / after the keyframe span.
        prefix = source_t < times[0]
        output[prefix] = source_t[prefix] / speeds[0]

        suffix = (
            (source_t > times[-1])
            if len(times) >= 2
            else np.zeros_like(source_t, dtype=bool)
        )
        output[suffix] = output_times[-1] + (source_t[suffix] - times[-1]) / speeds[-1]

        interior = ~prefix & ~suffix
        if len(times) == 1:
            output[interior] = (
                output_times[0] + (source_t[interior] - times[0]) / speeds[0]
            )
        else:
            for index in range(len(times) - 1):
                in_segment = interior & (source_t >= times[index])
                if index < len(times) - 2:
                    in_segment &= source_t < times[index + 1]
                t0, t1 = times[index], times[index + 1]
                v0, v1 = speeds[index], speeds[index + 1]
                tau0 = output_times[index]
                delta_s = source_t[in_segment] - t0
                if v0 == v1:
                    output[in_segment] = tau0 + delta_s / v0
                else:
                    # tau = tau0 + integral of 1/v(sigma), and v(sigma) is
                    # linear in source time, hence the closed-form logarithm.
                    speed_here = v0 + (v1 - v0) * delta_s / (t1 - t0)
                    output[in_segment] = tau0 + (
                        (t1 - t0) * np.log(speed_here / v0) / (v1 - v0)
                    )

        output = np.clip(output, 0.0, total_duration)
        return float(output) if scalar_input else output

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        self._mapping = self._build_mapping(clip.duration)
        self.output_duration = float(self._mapping[3])
        new_clip = clip.time_transform(self.source_time, apply_to=["mask", "audio"])
        return new_clip.with_duration(self.output_duration)
