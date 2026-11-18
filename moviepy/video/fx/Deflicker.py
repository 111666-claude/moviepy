from collections import OrderedDict
from dataclasses import dataclass
from numbers import Real
from threading import Lock

import numpy as np

from moviepy.Clip import Clip
from moviepy.Effect import Effect


class _BrightnessCache:
    """Bounded LRU cache of the mean brightness of a clip's frames.

    Only scalar brightness values are stored (never the frames themselves),
    so the memory used stays bounded by the window size, independently of
    the clip length.
    """

    def __init__(self, get_frame, fps, maxsize):
        self.get_frame = get_frame
        self.fps = fps
        self.maxsize = maxsize
        self._values = OrderedDict()
        self._lock = Lock()

    def brightness(self, index):
        """Return the mean brightness (0-255 scale) of frame ``index``."""
        with self._lock:
            value = self._values.get(index)
            if value is not None:
                self._values.move_to_end(index)
                return value

        # The underlying clip may itself cache frames, so fetching outside
        # the lock keeps it free while the (potentially expensive) frame is
        # being computed.
        frame = self.get_frame(index / self.fps)
        value = float(np.mean(frame, dtype=np.float64))

        with self._lock:
            self._values[index] = value
            self._values.move_to_end(index)
            while len(self._values) > self.maxsize:
                self._values.popitem(last=False)
        return value


@dataclass
class Deflicker(Effect):
    """Temporal deflickering of a video clip.

    Time-lapses shot in auto-exposure mode often flicker: the exposure
    jumps every few frames and the picture flashes. This effect estimates,
    for every frame, a multiplicative gain which pulls the frame's mean
    brightness toward the median brightness of the frames in a temporal
    window centered on it. The gain is clamped so that the correction can
    never exceed ``max_gain`` (relative deviation from 1.0).

    Parameters
    ----------
    radius : int
      Half-size of the temporal window, in frames. For each frame the
      brightness median is computed over the frames whose index is within
      ``radius`` of the current frame (clipped at the clip boundaries), so
      the window holds up to ``2 * radius + 1`` frames. Must be a positive
      integer.

    max_gain : float, optional
      Maximum allowed correction as a relative deviation from 1.0. The
      applied gain is clamped to ``[1 - max_gain, 1 + max_gain]``. For
      instance ``max_gain=0.1`` forbids boosting or dimming a frame by more
      than 10%. Must verify ``0 <= max_gain < 1``. Defaults to ``0.25``.

    scene_cut_threshold : float, optional
      Relative brightness jump between two consecutive frames above which
      the boundary is treated as a scene cut. At a scene cut, frames from
      the other side of the cut are excluded from the median window, so a
      genuine transition is never "corrected" away. Set it to ``0`` or
      ``None`` to disable scene cut detection. Must verify
      ``0 <= scene_cut_threshold <= 1``. Defaults to ``0.5``. A value above
      0.5 is needed only to catch very dark transitions; exposure flicker
      usually stays below this level.

    Notes
    -----
    - The clip must have an ``fps`` attribute (any ``VideoFileClip`` or
      clip created with an explicit fps does), as frame indices are derived
      from it.
    - Duration, fps, audio and mask of the clip are preserved unchanged;
      the mask is not deflickered.
    - Frames are never fully cached: only their scalar mean brightness is,
      in a bounded LRU cache, so random access and sequential export give
      the exact same result at any given time while memory stays bounded.
    - A frame with zero (or near zero, below ``1e-6`` on the 0-255 scale)
      brightness cannot be brightened safely and is left untouched.
    - Clips whose frame size varies over time are supported: brightness is
      computed per frame from whatever shape the frame has at that time.

    Example
    -------

    .. code:: python

        # Smooth the exposure using a window of 11 frames, never correcting
        # a frame by more than 15%.
        stable = flickery_clip.with_effects(
            [vfx.Deflicker(radius=5, max_gain=0.15)]
        )
    """

    radius: int
    max_gain: float = 0.25
    scene_cut_threshold: float = 0.5

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        if not isinstance(self.radius, int) or isinstance(self.radius, bool):
            raise TypeError("'radius' must be a positive integer")
        if self.radius < 1:
            raise ValueError("'radius' must be a positive integer")

        if not isinstance(self.max_gain, Real) or isinstance(self.max_gain, bool):
            raise TypeError("'max_gain' must be a real number")
        if not (0 <= float(self.max_gain) < 1):
            raise ValueError("'max_gain' must verify 0 <= max_gain < 1")

        if self.scene_cut_threshold is not None:
            if not isinstance(self.scene_cut_threshold, Real) or isinstance(
                self.scene_cut_threshold, bool
            ):
                raise TypeError("'scene_cut_threshold' must be a real number or None")
            if not (0 <= float(self.scene_cut_threshold) <= 1):
                raise ValueError(
                    "'scene_cut_threshold' must be between 0 and 1 inclusive"
                )

        fps = getattr(clip, "fps", None)
        if fps is None:
            raise ValueError(
                "Deflicker requires the clip to have an 'fps' attribute. "
                "Set one with clip.with_fps(fps) before applying the effect."
            )

        duration = getattr(clip, "duration", None)
        n_frames = None if duration is None else int(duration * fps)

        # The window spans at most 2 * radius + 1 indices. Cut detection may
        # inspect one extra index on each side, and an evicted entry would be
        # silently refetched, so size the cache for the full span plus two.
        cache = _BrightnessCache(
            get_frame=clip.get_frame,
            fps=fps,
            maxsize=2 * self.radius + 3,
        )

        detect_cuts = bool(self.scene_cut_threshold)
        lower_gain = 1.0 - float(self.max_gain)
        upper_gain = 1.0 + float(self.max_gain)

        def index_at(t):
            # Match MoviePy's own frame indexing convention (see
            # ``Clip.iter_frames`` and ``BitmapClip``): the frame displayed at
            # time t is the one of index int(t * fps). Rounding would fetch a
            # different frame than the one the gain is applied to.
            index = int(float(t) * fps)
            if n_frames is not None:
                index = min(max(index, 0), n_frames - 1)
            return max(index, 0)

        def is_cut(left_index, right_index):
            if not detect_cuts:
                return False
            left = cache.brightness(left_index)
            right = cache.brightness(right_index)
            reference = max(left, right)
            if reference < 1e-6:
                return False
            return abs(right - left) / reference > float(self.scene_cut_threshold)

        def window_bounds(center):
            """Return the inclusive ``(start, end)`` indices of the median
            window around ``center``, clipped at the clip boundaries and
            contracted on each side up to the nearest detected scene cut.
            """
            start = center - self.radius
            end = center + self.radius
            if n_frames is not None:
                start = max(start, 0)
                end = min(end, n_frames - 1)

            # Scan outward from the center. A cut between indices i and i+1
            # means frame i+1 belongs to another shot, so the left edge of
            # the window becomes i+1; symmetrically a cut at i-1/i makes i-1
            # the right edge. The nearest cut on each side is the binding one.
            for i in range(start, center):
                if is_cut(i, i + 1):
                    start = i + 1
            for i in range(end, center, -1):
                if is_cut(i - 1, i):
                    end = i - 1
            return start, end

        def filter(get_frame, t):
            center = index_at(t)
            start, end = window_bounds(center)

            window_brightness = np.array(
                [cache.brightness(i) for i in range(start, end + 1)]
            )
            target = float(np.median(window_brightness))

            # Read the frame once and derive its brightness from the same
            # array, so the gain always matches the frame it is applied to
            # even if the source clip rounds time indices differently.
            frame = get_frame(t)
            current = float(np.mean(frame, dtype=np.float64))

            if current < 1e-6 or target < 1e-6:
                gain = 1.0
            else:
                gain = np.clip(target / current, lower_gain, upper_gain)

            result = np.asarray(frame, dtype=np.float64) * gain
            result = np.clip(result, 0, 255)
            if np.issubdtype(frame.dtype, np.integer):
                result = np.round(result).astype(frame.dtype)
            return result

        new_clip = clip.transform(filter)
        # Exposed mainly for testing/inspection: the bounded brightness
        # cache backing the effect and its window-bounds computation.
        new_clip.deflicker_cache = cache
        new_clip.deflicker_window_bounds = window_bounds
        return new_clip
