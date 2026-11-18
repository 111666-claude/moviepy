import math
import numbers
from dataclasses import dataclass

import numpy as np

from moviepy.Clip import Clip
from moviepy.Effect import Effect


@dataclass
class ResampleFPS(Effect):
    """Returns a copy of the clip resampled to a new frame rate, with optional
    linear blending (interpolation) between adjacent source frames.

    Unlike ``Clip.with_fps``, which only changes the fps metadata and therefore
    duplicates or drops source frames, this effect builds a new frame function
    so that every output frame is computed from the source frame times:

    - with ``blend=True`` (default), each output frame at time ``t`` is a linear
      blend ``(1 - w) * frame[k] + w * frame[k + 1]`` of the two source frames
      surrounding ``t``, where ``k = floor(t * source_fps)`` and
      ``w = t * source_fps - k``. This gives smooth motion when converting e.g.
      24 fps footage to a 60 fps preview.
    - with ``blend=False``, output frames simply sample the source frame at
      their own time, which reproduces the classic duplicate/drop behavior of
      ``with_fps``.

    The same weights are applied to the clip's mask (if any). The audio, the
    duration and the start/end times of the clip are left untouched, and frame
    access stays a pure function of time, so random access and sequential
    export give identical results.

    Notes
    -----
    - The clip must have a defined ``fps`` (the source frame rate), otherwise
      a ``ValueError`` is raised.
    - Non-integer ratios (e.g. 24 -> 60 fps) are supported.
    - If the clip has a known duration, source frame indices are clamped to
      the last available frame, so the first and last output frames are exact
      copies of the first and last source frames. A single-frame clip simply
      repeats that frame.
    - If two adjacent frames to blend do not have the same shape (variable
      size frames), the nearest of the two is used for that output frame.

    Parameters
    ----------
    fps : float
      Target frame rate, must be a finite number strictly greater than 0.

    blend : bool, optional
      Whether to linearly blend adjacent source frames (default ``True``).
      If ``False``, plain nearest-frame resampling is performed.

    Examples
    --------
    >>> clip = VideoFileClip("movie_24fps.avi")
    >>> smooth_clip = clip.with_effects([vfx.ResampleFPS(60)])
    """

    fps: float = None
    blend: bool = True

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        fps = self.fps
        if (
            not isinstance(fps, numbers.Real)
            or isinstance(fps, bool)
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError(f"ResampleFPS requires a finite fps > 0, got {fps!r}")

        if getattr(clip, "fps", None) is None:
            raise ValueError(
                "ResampleFPS requires the clip to have a defined `fps` "
                "(the source frame rate). Set it with `clip.with_fps(...)` "
                "before applying the effect."
            )

        src_fps = clip.fps

        if clip.duration is not None:
            # Number of source frames, at least 1 (single-frame clip)
            n_frames = max(1, int(clip.duration * src_fps))
        else:
            n_frames = None

        def make_frame_function(get_frame):
            """Build a resampled frame function from a source ``get_frame``,
            shared by the clip and its mask so both use the same weights.
            """

            def frame_function(t):
                if not self.blend:
                    # Classic duplicate/drop resampling
                    return get_frame(t)

                pos = t * src_fps
                if pos <= 0:
                    return get_frame(0.0)

                k0 = int(math.floor(pos))
                weight = pos - k0

                if n_frames is not None:
                    last = n_frames - 1
                    if k0 >= last:
                        # First/last frames (and beyond) are exact copies
                        return get_frame(last / src_fps)

                if weight == 0:
                    return get_frame(k0 / src_fps)

                frame0 = get_frame(k0 / src_fps)
                frame1 = get_frame((k0 + 1) / src_fps)

                if frame0.shape != frame1.shape:
                    # Variable-size frames cannot be blended: use nearest
                    return frame0 if weight < 0.5 else frame1

                blended = (1.0 - weight) * frame0 + weight * frame1
                if frame0.dtype.kind in "iu":
                    info = np.iinfo(frame0.dtype)
                    blended = np.rint(blended).clip(info.min, info.max)
                return blended.astype(frame0.dtype)

            return frame_function

        new_clip = clip.copy()
        new_clip.frame_function = make_frame_function(clip.get_frame)
        new_clip.fps = fps

        if getattr(clip, "mask", None) is not None:
            new_mask = clip.mask.copy()
            new_mask.frame_function = make_frame_function(clip.mask.get_frame)
            new_mask.fps = fps
            new_clip.mask = new_mask

        # audio, duration, start and end are preserved by the shallow copy
        return new_clip
