"""Tests for the ``vfx.Perspective`` four-corner perspective mapping effect."""

import os

import numpy as np

import pytest

from moviepy import AudioArrayClip, ImageClip, VideoClip, vfx


WIDTH, HEIGHT = 100, 80
# A trapezoid fully inside the default canvas, with integer corner coordinates.
TRAPEZOID = ((10, 20), (90, 10), (95, 70), (5, 75))
# The four source-frame corners, in the same order as the target points.
SOURCE_CORNERS = ((0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT))


def make_color_block_clip(duration=1, fps=4):
    """A clip whose four quadrants have distinct solid colors."""

    def make_frame(t):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[: HEIGHT // 2, : WIDTH // 2] = (255, 0, 0)
        frame[: HEIGHT // 2, WIDTH // 2 :] = (0, 255, 0)
        frame[HEIGHT // 2 :, : WIDTH // 2] = (0, 0, 255)
        frame[HEIGHT // 2 :, WIDTH // 2 :] = (255, 255, 0)
        return frame

    return VideoClip(make_frame, duration=duration).with_fps(fps)


def make_corner_marker_clip():
    """A clip carrying a distinct color on each of its four corner pixels."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[0, 0] = (255, 0, 0)
    frame[0, WIDTH - 1] = (0, 255, 0)
    frame[HEIGHT - 1, WIDTH - 1] = (0, 0, 255)
    frame[HEIGHT - 1, 0] = (255, 255, 0)
    return ImageClip(frame).with_duration(1).with_fps(2)


def test_perspective_identity_mapping():
    """Mapping the corners onto themselves leaves frames and masks untouched."""
    clip = make_color_block_clip()
    result = clip.with_effects([vfx.Perspective(SOURCE_CORNERS)])

    assert result.size == clip.size
    assert np.array_equal(result.get_frame(0), clip.get_frame(0))

    mask = result.mask.get_frame(0)
    assert mask.shape == (HEIGHT, WIDTH)
    assert (mask == 1.0).all()


@pytest.mark.parametrize("resample", ["nearest", "bilinear"])
def test_perspective_output_canvas_and_transparency(resample):
    """Outside the target quadrilateral the frame is black and the mask is 0."""
    clip = make_color_block_clip()
    result = clip.with_effects(
        [vfx.Perspective(TRAPEZOID, resample=resample, canvas_size=(120, 100))]
    )

    assert tuple(result.size) == (120, 100)
    frame = result.get_frame(0)
    mask = result.mask.get_frame(0)

    assert frame.shape == (100, 120, 3)
    assert frame.dtype == np.uint8
    assert mask.shape == (100, 120)

    # A canvas corner is guaranteed to be outside the trapezoid.
    assert (frame[0, 0] == 0).all()
    assert mask[0, 0] == 0.0

    # The center of the warped quadrilateral stays opaque.
    center_x = sum(point[0] for point in TRAPEZOID) // 4
    center_y = sum(point[1] for point in TRAPEZOID) // 4
    assert mask[center_y, center_x] == 1.0
    # Some warped content must be visible inside the quadrilateral.
    assert frame[center_y, center_x].sum() > 0


def test_perspective_corner_alignment():
    """With nearest sampling, corner pixels land on the target corner points."""
    clip = make_corner_marker_clip()
    result = clip.with_effects(
        [vfx.Perspective(TRAPEZOID, resample="nearest", canvas_size=(WIDTH, HEIGHT))]
    )
    frame = result.get_frame(0)

    expected_colors = {
        (255, 0, 0): TRAPEZOID[0],  # top-left source corner
        (0, 255, 0): TRAPEZOID[1],  # top-right source corner
        (0, 0, 255): TRAPEZOID[2],  # bottom-right source corner
        (255, 255, 0): TRAPEZOID[3],  # bottom-left source corner
    }
    for color, (x, y) in expected_colors.items():
        # OpenCV only covers up to the last source pixel row/column, so a
        # boundary marker may be clipped entirely or shifted by the discrete
        # pixel grid. Markers that survive must stay within a couple of pixels
        # of their target corner.
        distances = [
            abs(px - x) + abs(py - y)
            for py in range(HEIGHT)
            for px in range(WIDTH)
            if (frame[py, px] == color).all()
        ]
        if distances:
            assert min(distances) <= 2

    # The top-left corner pixel is never clipped and maps exactly onto target.
    x, y = TRAPEZOID[0]
    assert (frame[y, x] == (255, 0, 0)).all()
    assert result.mask.get_frame(0)[y, x] == 1.0


def test_perspective_nearest_vs_bilinear():
    """Bilinear sampling softens edges while nearest keeps hard color blocks."""
    clip = make_color_block_clip()
    nearest = clip.with_effects(
        [vfx.Perspective(TRAPEZOID, resample="nearest", canvas_size=(WIDTH, HEIGHT))]
    ).get_frame(0)
    bilinear = clip.with_effects(
        [vfx.Perspective(TRAPEZOID, resample="bilinear", canvas_size=(WIDTH, HEIGHT))]
    ).get_frame(0)

    # Nearest sampling only reproduces the four original solid colors plus
    # the black background.
    unique_nearest = {tuple(color) for color in nearest.reshape(-1, 3)}
    assert unique_nearest <= {
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 0, 0),
    }

    # Bilinear sampling introduces blended colors along the warped edges.
    unique_bilinear = {tuple(color) for color in bilinear.reshape(-1, 3)}
    assert len(unique_bilinear) > len(unique_nearest)


def test_perspective_transforms_existing_mask():
    """A pre-existing mask undergoes the same mapping as the video frames."""
    clip = make_color_block_clip()
    mask = VideoClip(
        lambda t: np.full((HEIGHT, WIDTH), 0.37),
        is_mask=True,
        duration=1,
    ).with_fps(4)
    clip = clip.with_mask(mask)

    result = clip.with_effects(
        [vfx.Perspective(TRAPEZOID, resample="nearest", canvas_size=(120, 100))]
    )

    assert tuple(result.mask.size) == (120, 100)
    warped_mask = result.mask.get_frame(0)

    # Outside the quadrilateral the mask is transparent...
    assert warped_mask[0, 0] == 0.0

    # ...while the interior keeps its original semi-transparent value.
    center_x = sum(point[0] for point in TRAPEZOID) // 4
    center_y = sum(point[1] for point in TRAPEZOID) // 4
    assert warped_mask[center_y, center_x] == pytest.approx(0.37)


def test_perspective_auto_mask_is_polygon_geometry():
    """The generated mask matches the rasterized target quadrilateral."""
    import cv2

    clip = make_color_block_clip()
    result = clip.with_effects(
        [vfx.Perspective(TRAPEZOID, canvas_size=(WIDTH, HEIGHT))]
    )
    mask = result.mask.get_frame(0)

    grid_x, grid_y = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    centers = np.stack([grid_x + 0.5, grid_y + 0.5], axis=-1).reshape(-1, 2)
    target = np.array(TRAPEZOID, dtype=np.float32)
    inside = np.array(
        [cv2.pointPolygonTest(target, tuple(point), False) >= 0 for point in centers]
    ).reshape(HEIGHT, WIDTH)

    # Every disagreement must sit on the rasterized boundary of the polygon.
    distance = np.array(
        [abs(cv2.pointPolygonTest(target, tuple(point), True)) for point in centers]
    ).reshape(HEIGHT, WIDTH)
    disagreements = (mask > 0.5) != inside
    assert disagreements.sum() > 0  # anti-aliased boundary exists
    assert distance[disagreements].max() < 0.75


def test_perspective_on_mask_clip():
    """Applying the effect directly to a mask clip keeps it mask-only."""
    mask = VideoClip(
        lambda t: np.full((HEIGHT, WIDTH), 0.5),
        is_mask=True,
        duration=1,
    ).with_fps(2)
    result = mask.with_effects([vfx.Perspective(TRAPEZOID)])

    assert result.is_mask is True
    assert result.mask is None
    frame = result.get_frame(0)
    assert frame.ndim == 2
    assert frame.shape == (HEIGHT, WIDTH)
    # The warped value survives inside the quadrilateral.
    assert frame.max() == pytest.approx(0.5)


def test_perspective_uint8_mask_normalized():
    """A uint8 mask (e.g. loaded from an image) is warped in the 0-1 range."""
    mask_image = np.full((40, 50), 128, dtype=np.uint8)
    mask = ImageClip(mask_image, is_mask=True).with_duration(1)
    result = mask.with_effects(
        [vfx.Perspective(((5, 5), (45, 5), (45, 35), (5, 35)))]
    )

    frame = result.get_frame(0)
    assert np.issubdtype(frame.dtype, np.floating)
    assert frame[20, 25] == pytest.approx(128 / 255, abs=0.02)


def test_perspective_image_clip_without_fps():
    """Clips without an fps attribute (e.g. ImageClip) are handled."""
    clip = ImageClip(np.zeros((40, 50, 3), dtype=np.uint8)).with_duration(1)
    result = clip.with_effects(
        [vfx.Perspective(((5, 5), (45, 5), (45, 35), (5, 35)))]
    )

    assert tuple(result.size) == (50, 40)
    assert result.get_frame(0).shape == (40, 50, 3)
    assert result.mask is not None


def test_perspective_image_clip_rgba_mask():
    """A mask carried by an RGBA ImageClip keeps its semi-transparent values."""
    rgba = np.zeros((40, 50, 4), dtype=np.uint8)
    rgba[:, :, :3] = 100
    rgba[:, :, 3] = 64
    clip = ImageClip(rgba).with_duration(1)

    result = clip.with_effects(
        [vfx.Perspective(((5, 5), (45, 5), (45, 35), (5, 35)))]
    )
    assert result.mask.get_frame(0)[20, 25] == pytest.approx(64 / 255, abs=0.02)


def test_perspective_keeps_audio_fps_duration():
    """Audio, fps and duration are preserved through the transformation."""
    clip = make_color_block_clip(duration=0.5, fps=4)
    audio = AudioArrayClip(np.zeros((2205, 2), dtype=np.float32), fps=44100)
    clip = clip.with_audio(audio)

    result = clip.with_effects(
        [vfx.Perspective(TRAPEZOID, canvas_size=(120, 100))]
    )

    assert result.fps == 4
    assert result.duration == 0.5
    assert result.audio is not None
    assert result.audio.fps == 44100


def test_perspective_is_per_frame_without_caching():
    """Each frame is warped on demand and no full video is buffered."""
    calls = {"count": 0}

    def make_frame(t):
        calls["count"] += 1
        return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    clip = VideoClip(make_frame, duration=1).with_fps(4)
    result = clip.with_effects([vfx.Perspective(TRAPEZOID)])

    # Building the transformed clip probes t=0 once to determine its size;
    # every later read must trigger exactly one fresh source-frame evaluation.
    baseline = calls["count"]
    result.get_frame(0.25)
    result.get_frame(0.5)
    result.get_frame(0.75)
    assert calls["count"] - baseline == 3

    # Re-reading a frame goes back to the source instead of hitting a
    # precomputed buffer of the whole clip.
    result.get_frame(0.25)
    assert calls["count"] - baseline == 4

    # A long clip does not allocate a frame buffer proportional to its length.
    long_clip = VideoClip(make_frame, duration=3600).with_fps(30)
    long_result = long_clip.with_effects([vfx.Perspective(TRAPEZOID)])
    assert long_result.get_frame(1000).shape == (HEIGHT, WIDTH, 3)


def test_perspective_animated_points():
    """Target points can be defined as a function of time."""
    clip = make_color_block_clip()
    points = lambda t: (
        (10 + 20 * t, 20),
        (90, 10 + 10 * t),
        (95 - 10 * t, 70),
        (5, 75 - 10 * t),
    )
    result = clip.with_effects([vfx.Perspective(points, canvas_size=(120, 100))])

    first = result.get_frame(0)
    later = result.get_frame(0.5)
    assert first.shape == later.shape
    assert not np.array_equal(first, later)

    # The mask follows the moving quadrilateral.
    assert not np.array_equal(result.mask.get_frame(0), result.mask.get_frame(0.5))


def test_perspective_renders_to_file(util):
    """The transformed clip can be rendered and read back with the new size."""
    from moviepy import VideoFileClip

    clip = make_color_block_clip(duration=0.5, fps=4)
    result = clip.with_effects(
        [vfx.Perspective(TRAPEZOID, resample="nearest", canvas_size=(120, 100))]
    )
    output = os.path.join(util.TMP_DIR, "perspective.mp4")
    result.write_videofile(output, fps=4, logger=None, codec="libx264")

    with VideoFileClip(output) as rendered:
        assert tuple(rendered.size) == (120, 100)
        assert rendered.fps == 4
        assert rendered.duration == pytest.approx(0.5, abs=0.05)
        frame = rendered.get_frame(0)
        assert frame.shape == (100, 120, 3)
        assert (frame[0, 0] == 0).all()


def test_perspective_convenience_method():
    """``clip.perspectived`` is equivalent to the effect."""
    clip = make_color_block_clip()
    via_effect = clip.with_effects([vfx.Perspective(TRAPEZOID)])
    via_method = clip.perspectived(TRAPEZOID)

    assert via_method.size == via_effect.size
    assert np.array_equal(via_method.get_frame(0.25), via_effect.get_frame(0.25))


@pytest.mark.parametrize(
    ("points", "message"),
    (
        (((0, 0), (1, 1), (2, 2)), "exactly four"),
        (((0, 0), (1, 1, 2), (2, 2), (3, 3)), r"four \(x, y\)"),
        (((0, 0), (50, 0), (50, 40), (np.nan, 40)), "finite"),
        (((0, 0), (np.inf, 0), (50, 40), (0, 40)), "finite"),
        (((0, 0), (10, 0), (20, 0), (5, 20)), "collinear"),
        (((0, 0), (0, 0), (50, 40), (0, 40)), "collinear"),
        (((0, 0), (50, 40), (50, 0), (0, 40)), "convex"),
        (((0, 0), (50, 0), (10, 20), (0, 40)), "convex"),
    ),
)
def test_perspective_invalid_points(points, message):
    clip = make_color_block_clip()
    with pytest.raises(ValueError, match=message):
        clip.with_effects([vfx.Perspective(points)]).get_frame(0)


def test_perspective_invalid_points_from_callable():
    """Invalid points returned at render time also raise a clear error."""
    clip = make_color_block_clip()
    points = lambda t: TRAPEZOID if t < 0.25 else ((0, 0), (1, 1), (2, 2))
    result = clip.with_effects([vfx.Perspective(points)])

    result.get_frame(0)  # valid at t=0
    with pytest.raises(ValueError, match="exactly four"):
        result.get_frame(0.5)


@pytest.mark.parametrize(
    "canvas_size",
    ((100,), 100, (100.5, 40), (100, 0), (-10, 40), (0, 0)),
)
def test_perspective_invalid_canvas_size(canvas_size):
    clip = make_color_block_clip()
    with pytest.raises(ValueError, match="canvas_size"):
        clip.with_effects(
            [vfx.Perspective(TRAPEZOID, canvas_size=canvas_size)]
        ).get_frame(0)


def test_perspective_invalid_resample():
    clip = make_color_block_clip()
    with pytest.raises(ValueError, match="resample"):
        clip.with_effects(
            [vfx.Perspective(TRAPEZOID, resample="bicubic")]
        ).get_frame(0)


def test_rotate_effect_unchanged():
    """The pre-existing Rotate effect keeps working as before."""
    clip = make_color_block_clip()
    rotated = clip.with_effects([vfx.Rotate(90)])
    frame = rotated.get_frame(0)

    # A 90-degree rotation swaps width and height and transposes the frame.
    assert tuple(rotated.size) == (HEIGHT, WIDTH)
    assert frame.shape == (WIDTH, HEIGHT, 3)
    expected = np.transpose(clip.get_frame(0), axes=[1, 0, 2])[::-1]
    assert np.array_equal(frame, expected)
