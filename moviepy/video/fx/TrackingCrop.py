import numbers
from dataclasses import dataclass
from math import isfinite

import numpy as np

from moviepy.Clip import Clip
from moviepy.Effect import Effect


def _is_real_number(value):
    """True for real numeric values, including numpy scalars but not bools."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


@dataclass
class TrackingCrop(Effect):
    """Effect cropping a fixed-size window whose center follows time
    keyframes, e.g. to turn a widescreen game recording into a vertical
    short that keeps a moving character on screen.

    The crop window has a constant ``width`` x ``height`` so every output
    frame keeps the same dimensions. At each time ``t`` the window center is
    obtained by linearly interpolating the positions of the keyframes, then
    clamped so that the whole window always stays inside the source frame.
    Unlike the static ``Crop`` effect, the cropped rectangle is therefore a
    function of time.

    Parameters
    ----------

    width : int or float
        Width in pixels of the output clip. Must be greater than 0 and not
        greater than the width of the source clip.

    height : int or float
        Height in pixels of the output clip. Must be greater than 0 and not
        greater than the height of the source clip.

    keyframes : iterable of tuple
        Time-ordered center position keyframes, each one written as
        ``(time, x_center, y_center)``. ``time`` is in seconds and the
        coordinates are the center of the crop window in pixels inside the
        source frame. At least two keyframes are required. The times must be
        finite and strictly increasing, and every coordinate must be finite.
        Keyframes may lie outside the frame: the window is then shifted until
        it fully fits inside the frame. At times before the first keyframe or
        after the last one, the first/last keyframe position is held constant.

    Examples
    --------

    Track a character moving from left to right, keeping a 270x480 vertical
    window on it:

    >>> TrackingCrop(
    ...     width=270,
    ...     height=480,
    ...     keyframes=[(0, 100, 240), (2, 600, 240), (4, 1180, 240)],
    ... )

    The same effect through the clip method:

    >>> clip.with_effects(
    ...     [
    ...         TrackingCrop(
    ...             width=270,
    ...             height=480,
    ...             keyframes=[(0, 100, 240), (4, 1180, 240)],
    ...         )
    ...     ]
    ... )
    """

    width: float
    height: float
    keyframes: list

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        if not clip.has_constant_size:
            raise ValueError(
                "TrackingCrop requires a clip with a constant frame size, but "
                "the given clip has frames of varying dimensions."
            )

        src_width, src_height = clip.size

        if not _is_real_number(self.width) or not _is_real_number(self.height):
            raise ValueError(
                "TrackingCrop 'width' and 'height' must be numbers, got "
                f"width={self.width!r}, height={self.height!r}."
            )
        if not isfinite(self.width) or not isfinite(self.height):
            raise ValueError(
                "TrackingCrop 'width' and 'height' must be finite numbers, got "
                f"width={self.width!r}, height={self.height!r}."
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                "TrackingCrop 'width' and 'height' must be greater than 0, got "
                f"width={self.width!r}, height={self.height!r}."
            )
        if self.width > src_width or self.height > src_height:
            raise ValueError(
                "TrackingCrop crop window "
                f"({int(self.width)}x{int(self.height)}) is larger than the "
                f"source clip ({src_width}x{src_height}); the window must fit "
                "inside every source frame."
            )

        times, x_centers, y_centers = self._parse_keyframes()

        # Integer window dimensions. Slices use truncation like Crop does.
        crop_width = int(self.width)
        crop_height = int(self.height)

        # Reachable center range so that the whole window stays in the frame.
        min_x = crop_width / 2
        max_x = src_width - crop_width / 2
        min_y = crop_height / 2
        max_y = src_height - crop_height / 2

        def window_for_time(t):
            """Return the (x1, y1) top-left corner of the window at time t."""
            # np.interp clamps to the end values outside the keyframe time
            # range, i.e. boundary times hold the first/last keyframe.
            x_center = float(np.interp(t, times, x_centers))
            y_center = float(np.interp(t, times, y_centers))
            x_center = min(max(x_center, min_x), max_x)
            y_center = min(max(y_center, min_y), max_y)
            return (
                int(x_center - crop_width / 2),
                int(y_center - crop_height / 2),
            )

        expected_shape = (src_height, src_width)

        def filter(get_frame, t):
            frame = get_frame(t)
            if frame.shape[:2] != expected_shape:
                raise ValueError(
                    "TrackingCrop encountered a frame of size "
                    f"{frame.shape[1]}x{frame.shape[0]} but every source frame "
                    f"must be {src_width}x{src_height}; refusing to build a "
                    "clip with varying output dimensions."
                )
            x1, y1 = window_for_time(t)
            return frame[y1 : y1 + crop_height, x1 : x1 + crop_width]

        # apply_to=["mask"] crops the mask with the exact same window, so the
        # transparency stays aligned with the RGB frames. Duration, fps and
        # audio are left untouched by transform(keep_duration=True).
        return clip.transform(filter, apply_to=["mask"], keep_duration=True)

    def _parse_keyframes(self):
        """Validate the keyframes and return parallel lists of times/centers."""
        if self.keyframes is None:
            raise ValueError("TrackingCrop 'keyframes' must not be None.")
        try:
            keyframes = list(self.keyframes)
        except TypeError as exc:
            raise ValueError(
                "TrackingCrop 'keyframes' must be an iterable of "
                "(time, x_center, y_center) tuples."
            ) from exc

        if len(keyframes) < 2:
            raise ValueError(
                "TrackingCrop needs at least 2 keyframes to interpolate "
                f"between, got {len(keyframes)}."
            )

        times = []
        x_centers = []
        y_centers = []
        previous_time = None
        for index, keyframe in enumerate(keyframes):
            try:
                time, x_center, y_center = keyframe
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "TrackingCrop keyframe #%d is %r but every keyframe must "
                    "be a (time, x_center, y_center) tuple of 3 numbers."
                    % (index, keyframe)
                ) from exc

            values = (time, x_center, y_center)
            if not all(_is_real_number(value) for value in values):
                raise ValueError(
                    "TrackingCrop keyframe #%d is %r but time and center "
                    "coordinates must all be real numbers (booleans are not "
                    "accepted)." % (index, keyframe)
                )
            if not all(isfinite(value) for value in values):
                raise ValueError(
                    "TrackingCrop keyframe #%d is %r but contains a non-finite "
                    "value (NaN or infinity)." % (index, keyframe)
                )
            if time < 0:
                raise ValueError(
                    "TrackingCrop keyframe #%d has a negative time %r; times "
                    "must be non-negative." % (index, time)
                )
            if previous_time is not None and time <= previous_time:
                raise ValueError(
                    "TrackingCrop keyframe times must be strictly increasing, "
                    "but keyframe #%d time %r is not greater than the previous "
                    "time %r (duplicated or reversed times are not allowed)."
                    % (index, time, previous_time)
                )

            times.append(float(time))
            x_centers.append(float(x_center))
            y_centers.append(float(y_center))
            previous_time = time

        return (
            np.array(times),
            np.array(x_centers),
            np.array(y_centers),
        )
