"""Trajectory-based translation video stabilization effect."""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from moviepy.Clip import Clip
from moviepy.Effect import Effect


@dataclass
class Stabilize(Effect):
    """Stabilize a shaky clip by compensating a measured 2D camera trajectory.

    The input trajectory describes the camera translation (in pixels) at given
    times. It is smoothed with a centered moving average and the difference
    between the raw and smoothed trajectories is applied to every frame as a
    counter-translation, removing the high-frequency jitter while keeping the
    low-frequency camera motion.

    To guarantee a fixed-size output without any black border, each frame is
    cropped by a fixed margin on every side. The margin is either computed
    automatically from the largest compensation ever needed or provided
    explicitly through ``border_crop``. The actual crop parameters are stored
    on the effect instance after ``apply`` and returned alongside the clip.

    Parameters
    ----------
    trajectory : sequence
        The measured camera trajectory as a sequence of ``(time, dx, dy)``
        keyframes. ``time`` is in seconds, ``(dx, dy)`` is the camera
        translation in pixels (using the OpenCV convention: positive ``dx``
        shifts the image to the right, positive ``dy`` shifts it downwards).
        Missing keyframes are linearly interpolated, so any arbitrary time can
        be evaluated deterministically. The keyframes must cover the whole clip
        duration (``time[0] <= 0`` and ``time[-1] >= clip.duration``).
    smoothing : float, optional
        Width in seconds of the centered moving-average window used to smooth
        the trajectory. Larger values remove more motion. ``0`` disables
        smoothing, in which case every frame is fully compensated back to its
        original position. Default is ``1.0`` second.
    border_crop : tuple, optional
        Explicit fixed crop margins ``(left, right, top, bottom)`` in pixels
        applied on every frame. When ``None`` (default), the smallest margins
        that keep every compensation inside the croppable area are computed
        automatically. Margins are always integers and each one must be
        strictly smaller than the corresponding clip dimension.
    fps : int, optional
        Frame rate used to sample the trajectory while smoothing. Defaults to
        the clip ``fps``; it is only required when the clip has no ``fps``.
    interpolation : str, optional
        OpenCV interpolation used when warping frames: ``"nearest"``,
        ``"bilinear"`` or ``"bicubic"``. Default is ``"bilinear"``.

    Attributes
    ----------
    crop_margins : tuple or None
        After ``apply``, the actual margins used as
        ``(left, right, top, bottom)``.
    output_size : tuple or None
        After ``apply``, the ``(width, height)`` of the stabilized clip.
    smoothed_trajectory : np.ndarray or None
        After ``apply``, the smoothed trajectory sampled at the clip frame
        times, as an ``(N, 3)`` array of ``(time, dx, dy)`` rows.

    The returned clip also exposes the same values as the
    ``stabilize_crop_margins``, ``stabilize_output_size`` and
    ``stabilize_smoothed_trajectory`` attributes, so they stay reachable when
    the effect is applied through ``clip.with_effects([...])`` (which applies
    a copy of the effect).

    Examples
    --------
    Smooth a trajectory measured externally and stabilize a clip:

    .. code:: python

        trajectory = [(0, 0, 0), (0.5, 12, -3), (1.0, 4, 8), (1.5, -6, 2)]
        effect = vfx.Stabilize(trajectory, smoothing=0.5)
        stabilized = clip.with_effects([effect])
        print(effect.crop_margins, effect.output_size)

    Raises
    ------
    ValueError
        If the trajectory is empty, contains non-finite coordinates, is not
        ordered by strictly increasing time, does not cover the clip duration,
        has rows that are not 3-element ``(time, dx, dy)`` triples, or if a
        compensation cannot fit inside the croppable area.
    """

    trajectory: Sequence[Tuple[float, float, float]]
    smoothing: float = 1.0
    border_crop: Optional[Tuple[int, int, int, int]] = None
    fps: Optional[float] = None
    interpolation: str = "bilinear"

    # Populated by ``apply``, see the class docstring.
    crop_margins: Optional[Tuple[int, int, int, int]] = field(
        default=None, init=False, repr=False
    )
    output_size: Optional[Tuple[int, int]] = field(default=None, init=False, repr=False)
    smoothed_trajectory: Optional[np.ndarray] = field(
        default=None, init=False, repr=False
    )

    def _validate_trajectory(self, clip_duration):
        """Validate the raw keyframes and return them as a float array."""
        if self.trajectory is None or len(self.trajectory) == 0:
            raise ValueError("'trajectory' must contain at least one keyframe")

        try:
            raw = np.asarray(list(self.trajectory), dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                "'trajectory' must be a sequence of (time, dx, dy) triples"
            )

        if raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError(
                "'trajectory' must contain (time, dx, dy) triples, got shape"
                f" {raw.shape}"
            )

        if not np.isfinite(raw).all():
            raise ValueError(
                "'trajectory' contains non-finite (NaN or infinite) coordinates"
            )

        if np.any(np.diff(raw[:, 0]) <= 0):
            raise ValueError("'trajectory' times must be strictly increasing")

        if clip_duration is not None and clip_duration > 0 and raw.shape[0] < 2:
            raise ValueError(
                "'trajectory' needs at least two keyframes to cover a clip"
                " with a positive duration"
            )

        if clip_duration is not None and clip_duration > 0:
            if raw[0, 0] > 0 or raw[-1, 0] < clip_duration:
                raise ValueError(
                    "'trajectory' must cover the whole clip duration: expected"
                    f" keyframes from t<=0 to t>={clip_duration}, got"
                    f" [{raw[0, 0]}, {raw[-1, 0]}]"
                )

        if self.smoothing < 0:
            raise ValueError("'smoothing' must be a non-negative number of seconds")

        return raw

    def _sample_smoothed_trajectory(self, raw, duration, fps):
        """Linearly resample the trajectory and smooth dx/dy with a moving
        average. Returns an (N, 3) array of (time, smoothed dx, smoothed dy).
        """
        if duration is None:
            raise ValueError("The clip must have a duration to be stabilized")
        if fps is None:
            raise ValueError("'fps' is required when the clip has no fps attribute")
        if fps <= 0:
            raise ValueError("'fps' must be strictly positive")

        n_frames = max(1, int(round(duration * fps)))
        # Frame times stay inside [0, duration], the last keyframe covers the
        # end of the clip through linear interpolation.
        sample_times = np.arange(n_frames, dtype=float) / fps
        sample_times = np.minimum(sample_times, duration)

        sampled = np.empty((n_frames, 3), dtype=float)
        sampled[:, 0] = sample_times
        sampled[:, 1] = np.interp(sample_times, raw[:, 0], raw[:, 1])
        sampled[:, 2] = np.interp(sample_times, raw[:, 0], raw[:, 2])

        if self.smoothing == 0 or n_frames == 1:
            # No smoothing: the desired camera path is completely flat, locked
            # on the first frame. The whole raw motion becomes jitter to
            # compensate.
            smoothed = sampled.copy()
            smoothed[:, 1] = sampled[0, 1]
            smoothed[:, 2] = sampled[0, 2]
            return smoothed

        window = max(1, int(round(self.smoothing * fps)))

        def moving_average(values):
            # Centered boxcar filter with truncated windows at the edges, so
            # the smoothed curve keeps the same length and stays finite.
            cumulative = np.cumsum(np.insert(values, 0, 0.0))
            indices = np.arange(values.size)
            starts = np.maximum(indices - window + 1, 0)
            ends = np.minimum(indices + window, values.size)
            window_sums = cumulative[ends] - cumulative[starts]
            return window_sums / (ends - starts).astype(float)

        smoothed = sampled.copy()
        smoothed[:, 1] = moving_average(sampled[:, 1])
        smoothed[:, 2] = moving_average(sampled[:, 2])

        # Anchor the smoothed curve on the first frame so the compensation
        # (smoothed minus raw) is exactly zero at t=0: the first frame is kept
        # in place and only frame-to-frame jitter is removed.
        smoothed[:, 1] -= smoothed[0, 1] - sampled[0, 1]
        smoothed[:, 2] -= smoothed[0, 2] - sampled[0, 2]
        return smoothed

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        try:
            interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
            }[self.interpolation]
        except KeyError:
            raise ValueError(
                "'interpolation' must be either 'nearest', 'bilinear' or" " 'bicubic'"
            )

        fps = self.fps if self.fps is not None else getattr(clip, "fps", None)
        raw = self._validate_trajectory(clip.duration)
        smoothed = self._sample_smoothed_trajectory(raw, clip.duration, fps)
        self.smoothed_trajectory = smoothed

        w, h = clip.size

        # Compensation applied to each frame: smoothed (low-frequency) motion
        # minus the raw measured motion. The smoothed and raw curves share the
        # same anchor at t=0, so the first frame is not shifted and only the
        # frame-to-frame jitter is cancelled. With smoothing disabled the
        # smoothed curve is flat, giving a full lock on the first frame.
        raw_at_frames = np.empty_like(smoothed)
        raw_at_frames[:, 0] = smoothed[:, 0]
        raw_at_frames[:, 1] = np.interp(smoothed[:, 0], raw[:, 0], raw[:, 1])
        raw_at_frames[:, 2] = np.interp(smoothed[:, 0], raw[:, 0], raw[:, 2])
        compensation = smoothed - raw_at_frames

        max_dx = float(np.max(np.abs(compensation[:, 1])))
        max_dy = float(np.max(np.abs(compensation[:, 2])))

        if self.border_crop is None:
            # Smallest integer margins that keep every frame fully covered.
            left = right = int(np.ceil(max_dx - 1e-9))
            top = bottom = int(np.ceil(max_dy - 1e-9))
        else:
            if len(self.border_crop) != 4:
                raise ValueError(
                    "'border_crop' must be a (left, right, top, bottom) tuple"
                )
            left, right, top, bottom = (int(v) for v in self.border_crop)
            if min(left, right, top, bottom) < 0:
                raise ValueError("'border_crop' margins must be non-negative")
            # A one-pixel slack absorbs floating point rounding errors.
            if max(left, right) + 1.0 > w or max(top, bottom) + 1.0 > h:
                raise ValueError(
                    "'border_crop' margins exceed the clip size"
                    f" (clip size is {w}x{h})"
                )

        out_w, out_h = w - left - right, h - top - bottom
        if out_w <= 0 or out_h <= 0:
            raise ValueError(
                "The crop margins leave no usable frame area: clip size is"
                f" {w}x{h} and margins are {(left, right, top, bottom)}"
            )

        # Every output pixel must map back inside the source frame for every
        # possible compensation, otherwise a black (or replicated) border would
        # appear in a fixed-size output.
        if max_dx > max(left, right) + 1e-6 or max_dy > max(top, bottom) + 1e-6:
            raise ValueError(
                "The stabilization compensation exceeds the croppable area:"
                f" needs at least {int(np.ceil(max_dx))} horizontal and"
                f" {int(np.ceil(max_dy))} vertical margin pixels, got"
                f" {(left, right, top, bottom)}"
            )

        self.crop_margins = (left, right, top, bottom)
        self.output_size = (out_w, out_h)

        raw_times = raw[:, 0]
        raw_dx, raw_dy = raw[:, 1], raw[:, 2]

        def shift_at(t):
            """Residual camera motion (sx, sy) at arbitrary time t.

            The content must be moved by the opposite of this residual to
            cancel the jitter (see ``filter``).
            """
            raw_x = float(np.interp(t, raw_times, raw_dx))
            raw_y = float(np.interp(t, raw_times, raw_dy))
            smooth_x = float(np.interp(t, smoothed[:, 0], smoothed[:, 1]))
            smooth_y = float(np.interp(t, smoothed[:, 0], smoothed[:, 2]))
            return raw_x - smooth_x, raw_y - smooth_y

        def filter(get_frame, t):
            frame = get_frame(t)
            sx, sy = shift_at(t)
            # A camera shift of (dx, dy) records source point p at
            # p + (dx, dy). Cancelling the residual means moving the image
            # content by (-sx, -sy), and cropping the margins moves it by an
            # extra (-left, -top). OpenCV's warpAffine with translation
            # entries (tx, ty) moves content to (x + tx, y + ty), hence the
            # entries below. With the validated margins every sampled
            # coordinate stays inside the source frame, so no border pixels
            # can ever appear in the fixed-size output.
            matrix = np.float32([[1, 0, -(left + sx)], [0, 1, -(top + sy)]])
            is_mask = frame.ndim == 2
            return cv2.warpAffine(
                frame,
                matrix,
                (out_w, out_h),
                flags=interpolation if not is_mask else cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REPLICATE,
            )

        # ``apply_to=["mask"]`` warps the mask with the same per-frame spatial
        # transform. Audio, duration and fps are left untouched.
        new_clip = clip.transform(filter, apply_to=["mask"], keep_duration=True)

        # ``transform`` does not update the size metadata itself; the output
        # frames have the cropped size, so keep the clip attributes in sync.
        new_clip.size = (out_w, out_h)
        if new_clip.mask is not None:
            new_clip.mask.size = (out_w, out_h)

        # Expose the actual crop parameters directly on the clip as well, so
        # they remain available even though ``with_effects`` applies a copy of
        # the effect rather than this instance.
        new_clip.stabilize_crop_margins = self.crop_margins
        new_clip.stabilize_output_size = self.output_size
        new_clip.stabilize_smoothed_trajectory = self.smoothed_trajectory
        return new_clip
