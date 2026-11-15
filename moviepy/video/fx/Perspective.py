from dataclasses import dataclass

import cv2
import numpy as np

from moviepy.Clip import Clip
from moviepy.Effect import Effect


@dataclass
class Perspective(Effect):
    """Applies a four-corner perspective (homography) mapping to each frame.

    The rectangular content of every frame is projected onto an arbitrary
    convex target quadrilateral defined by four destination corner points.
    Unlike ``Rotate``, which only performs affine transforms, this effect
    can make a flat clip look like it is pasted onto a tilted surface (for
    example a billboard filmed at an angle).

    The four source corners of a frame of size ``(width, height)`` are, in
    order, ``(0, 0)`` (top-left), ``(width, 0)`` (top-right),
    ``(width, height)`` (bottom-right) and ``(0, height)`` (bottom-left).
    Each one is mapped onto the matching point in ``points``.

    Parameters
    ----------

    points : tuple or list or callable
        Four target corner points ``((x1, y1), (x2, y2), (x3, y3), (x4, y4))``
        corresponding, in order, to the top-left, top-right, bottom-right and
        bottom-left corners of the source frame. Coordinates are in pixels and
        may be negative or float. It can also be a function ``t -> points``
        to animate the mapping over time.

    resample : str, optional
        Sampling filter used when warping the image. One of ``"nearest"`` or
        ``"bilinear"``. Default is ``"bilinear"``.

    canvas_size : tuple, optional
        Size of the output clip as ``(width, height)``. By default, the output
        keeps the size of the input clip, so target points outside of the
        source canvas are cropped.

    Examples
    --------

    .. code:: python

        # Paste a 320x240 clip onto a trapezoid
        clip = clip.with_effects([
            vfx.Perspective(
                points=((40, 60), (280, 30), (300, 220), (20, 200)),
                canvas_size=(320, 240),
            )
        ])

        # Animate the trapezoid over time
        clip = clip.with_effects([
            vfx.Perspective(points=lambda t: quadrilateral_at(t))
        ])
    """

    points: object
    resample: str = "bilinear"
    canvas_size: tuple = None

    @staticmethod
    def _validate_points(points):
        """Validate the target quadrilateral and return it as a float array."""
        try:
            array = np.array(points, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "'points' must contain four (x, y) corner coordinates"
            ) from exc

        if array.shape != (4, 2):
            raise ValueError(
                "'points' must contain exactly four (x, y) corner coordinates, "
                f"got an array of shape {array.shape}"
            )

        if not np.isfinite(array).all():
            raise ValueError(
                "'points' must contain finite (x, y) coordinates; got NaN or "
                "infinite values"
            )

        # The four points must not be collinear (three aligned points make the
        # perspective mapping singular).
        for i in range(4):
            p0, p1, p2 = array[i], array[(i + 1) % 4], array[(i + 2) % 4]
            if np.linalg.det(np.vstack((p1 - p0, p2 - p0))) == 0:
                raise ValueError(
                    "'points' must define a non-degenerate quadrilateral; at "
                    "least three points are collinear"
                )

        # Reject self-intersecting (bow-tie) or concave quadrilaterals. The
        # signed cross products of consecutive edges must all share a sign.
        cross_products = []
        for i in range(4):
            edge_a = array[(i + 1) % 4] - array[i]
            edge_b = array[(i + 2) % 4] - array[(i + 1) % 4]
            cross_products.append(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0])
        if not (
            all(c > 0 for c in cross_products)
            or all(c < 0 for c in cross_products)
        ):
            raise ValueError(
                "'points' must define a convex quadrilateral; self-intersecting "
                "or concave point sets are not supported"
            )

        return array

    def _validate_canvas_size(self, clip_size):
        if self.canvas_size is None:
            return clip_size

        if (
            not isinstance(self.canvas_size, (tuple, list))
            or len(self.canvas_size) != 2
        ):
            raise ValueError(
                "'canvas_size' must be a (width, height) tuple, got "
                f"{self.canvas_size!r}"
            )

        width, height = self.canvas_size
        if (
            not isinstance(width, (int, np.integer))
            or not isinstance(height, (int, np.integer))
            or width <= 0
            or height <= 0
        ):
            raise ValueError(
                "'canvas_size' must contain two positive integers, got "
                f"{self.canvas_size!r}"
            )

        return (int(width), int(height))

    def _homography(self, source_points, target_points):
        """Build the source -> target homography and ensure it is invertible.

        ``cv2.findHomography`` works with corner coordinates while OpenCV's
        warping functions index pixels by their centers, so the homography is
        adjusted by a half-pixel translation on both sides. Without this step
        every mapped feature would be shifted by up to one pixel.
        """
        homography, _ = cv2.findHomography(source_points, target_points, method=0)
        if homography is None:
            raise ValueError(
                "The perspective transform defined by 'points' is not invertible"
            )

        determinant = np.linalg.det(homography)
        try:
            np.linalg.inv(homography)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "The perspective transform defined by 'points' is not invertible"
            ) from exc

        if not np.isfinite(determinant) or determinant == 0:
            raise ValueError(
                "The perspective transform defined by 'points' is not invertible"
            )

        half_pixel_shift = np.array(
            [[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], dtype=np.float64
        )
        homography = (
            half_pixel_shift @ homography @ np.linalg.inv(half_pixel_shift)
        )
        return homography.astype(np.float32)

    def _warp_frame(self, frame, homography, canvas_size):
        """Warp a single RGB frame or mask onto the target canvas.

        MoviePy masks use float values in ``[0, 1]``; uint8 masks (for example
        those built from images) are converted before warping so their values
        keep the same meaning afterwards.
        """
        is_mask = frame.ndim == 2
        if is_mask and np.issubdtype(frame.dtype, np.integer):
            frame = frame.astype(float) / 255.0

        border_value = 0 if is_mask else (0, 0, 0)
        return cv2.warpPerspective(
            frame,
            homography,
            canvas_size,
            flags=self.resample_flag,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        try:
            self.resample_flag = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
            }[self.resample]
        except KeyError:
            raise ValueError(
                "'resample' argument must be either 'nearest' or 'bilinear'"
            )

        canvas_size = self._validate_canvas_size(clip.size)
        width, height = clip.size

        source_points = np.array(
            [(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float32
        )

        if hasattr(self.points, "__call__"):
            get_points = self.points
        else:
            # Validate static points once, up front.
            self._validate_points(self.points)
            get_points = lambda t: self.points

        def filter(get_frame, t):
            target_points = self._validate_points(get_points(t))
            homography = self._homography(source_points, target_points)
            return self._warp_frame(get_frame(t), homography, canvas_size)

        new_clip = clip.transform(filter, apply_to=["mask"], keep_duration=True)

        # Without an explicit mask, generate one so the area outside of the
        # target quadrilateral is transparent on the output canvas. A mask clip
        # carries its own grayscale values and must not be given a sub-mask.
        if clip.mask is None and not getattr(clip, "is_mask", False):
            from moviepy.video.VideoClip import VideoClip

            def mask_frame(t):
                target_points = self._validate_points(get_points(t))
                homography = self._homography(source_points, target_points)
                return self._warp_frame(
                    np.ones((height, width), dtype=float),
                    homography,
                    canvas_size,
                )

            polygon_mask = VideoClip(is_mask=True, frame_function=mask_frame)
            if clip.duration is not None:
                polygon_mask = polygon_mask.with_duration(clip.duration)
            if getattr(clip, "fps", None) is not None:
                polygon_mask = polygon_mask.with_fps(clip.fps)
            new_clip.mask = polygon_mask

        return new_clip
