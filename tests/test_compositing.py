"""Compositing tests for use with pytest."""

import os

import numpy as np
from PIL import Image

import pytest

from moviepy import *
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip


# Common setup for the HUD-heavy compositing tests below: 12 semi-transparent
# 96x96 overlays on a 256x256 frame, rendered over 120 frames.
HUD_GRID_SIZE = (256, 256)
HUD_COUNT = 12
HUD_SIZE = (96, 96)
HUD_ALPHA = 128
HUD_FRAMES = 120
HUD_FPS = 30
# Sum of all pixels of the final frame (R+G+B) for the HUD layout below.
HUD_EXPECTED_FRAME_PIXEL_SUM = 26771616


def hud_clips():
    """Build the 12 semi-transparent overlays used by the HUD tests."""
    overlays = []
    for i in range(HUD_COUNT):
        clip = ColorClip(HUD_SIZE, color=(255, 255, 255, HUD_ALPHA))
        pos = (
            (i * 37) % (HUD_GRID_SIZE[0] - HUD_SIZE[0]),
            (i * 53) % (HUD_GRID_SIZE[1] - HUD_SIZE[1]),
        )
        overlays.append(clip.with_position(pos))
    return overlays


def opaque_composite():
    """A 256x256 opaque black background with the HUD overlays on top."""
    bg = ColorClip(HUD_GRID_SIZE, color=(0, 0, 0))
    return CompositeVideoClip([bg] + hud_clips(), use_bgclip=True)


def transparent_composite():
    """A transparent 256x256 canvas with the HUD overlays on top."""
    return CompositeVideoClip(hud_clips(), size=HUD_GRID_SIZE)


class ClipPixelTest:
    ALLOWABLE_COLOR_VARIATION = 3  # from 0-767: how much mismatch do we allow

    def __init__(self, clip):
        self.clip = clip

    def expect_color_at(self, ts, expected, xy=[0, 0]):
        frame = self.clip.frame_function(ts)
        r, g, b = expected
        actual = frame[xy[1]][xy[0]]
        diff = abs(actual[0] - r) + abs(actual[1] - g) + abs(actual[2] - b)

        mismatch = diff > ClipPixelTest.ALLOWABLE_COLOR_VARIATION
        assert (
            not mismatch
        ), "Expected (%02x,%02x,%02x) but got (%02x,%02x,%02x) at timestamp %s" % (
            *expected,
            *actual,
            ts,
        )


def test_clips_array(util):
    red = ColorClip((1024, 800), color=(255, 0, 0))
    green = ColorClip((1024, 800), color=(0, 255, 0))
    blue = ColorClip((1024, 800), color=(0, 0, 255))

    video = clips_array([[red, green, blue]])

    with pytest.raises(ValueError):  # duration not set
        video.with_effects([vfx.Resize(width=480)]).write_videofile(
            os.path.join(util.TMP_DIR, "test_clips_array.mp4")
        )


def test_clips_array_duration(util):
    filename = os.path.join(util.TMP_DIR, "test_clips_array.mp4")

    # NOTE: anyone knows what behaviour this sets ? If yes please replace
    # this comment.
    red = ColorClip((256, 200), color=(255, 0, 0))
    green = ColorClip((256, 200), color=(0, 255, 0))
    blue = ColorClip((256, 200), color=(0, 0, 255))

    video = clips_array([[red, green, blue]]).with_duration(5)
    with pytest.raises(AttributeError):  # fps not set
        video.write_videofile(filename)

    # this one should work correctly
    red.fps = green.fps = blue.fps = 30
    video = clips_array([[red, green, blue]]).with_duration(5)
    video.write_videofile(filename)


def test_concatenate_self(util):
    clip = BitmapClip([["AAA", "BBB"], ["CCC", "DDD"]], fps=1)
    target = BitmapClip([["AAA", "BBB"], ["CCC", "DDD"]], fps=1)

    concatenated = concatenate_videoclips([clip])

    concatenated.write_videofile(
        os.path.join(util.TMP_DIR, "test_concatenate_self.mp4")
    )
    assert concatenated == target


def test_concatenate_floating_point(util):
    """
    >>> print("{0:.20f}".format(1.12))
    1.12000000000000010658

    This test uses duration=1.12 to check that it still works when the clip
    duration is represented as being bigger than it actually is. Fixed in #1195.
    """
    clip = ColorClip([100, 50], color=[255, 128, 64], duration=1.12).with_fps(25.0)
    concat = concatenate_videoclips([clip])
    concat.write_videofile(os.path.join(util.TMP_DIR, "concat.mp4"), preset="ultrafast")


def test_blit_with_opacity():
    # has one second R, one second G, one second B
    size = (2, 2)
    clip1 = (
        ColorClip(size, color=(255, 0, 0), duration=1)
        + ColorClip(size, color=(0, 255, 0), duration=1)
        + ColorClip(size, color=(0, 0, 255), duration=1)
    )

    # overlay green at half opacity during first 2 sec
    clip2 = ColorClip(size, color=(0, 255, 0), duration=2).with_opacity(0.5)
    composite = CompositeVideoClip([clip1, clip2])
    bt = ClipPixelTest(composite)

    # red + 50% green
    bt.expect_color_at(0.5, (0x7F, 0x7F, 0x00))
    # green + 50% green
    bt.expect_color_at(1.5, (0x00, 0xFF, 0x00))
    # blue is after 2s, so keep untouched
    bt.expect_color_at(2.5, (0x00, 0x00, 0xFF))


def test_compositing_masks(util):
    # Has one R 30%, one G 30%, one B 30%
    clip1 = ColorClip((100, 100), (255, 0, 0, 76.5)).with_duration(2)
    clip2 = ColorClip((50, 50), (0, 255, 0, 76.5)).with_duration(2)
    clip3 = ColorClip((25, 25), (0, 0, 255, 76.5)).with_duration(2)

    compostite_clip1 = CompositeVideoClip(
        [clip1, clip2.with_position(("center", "center"))]
    )
    compostite_clip2 = CompositeVideoClip(
        [compostite_clip1, clip3.with_position(("center", "center"))]
    )

    # Load output file and check transparency
    frame = compostite_clip2.mask.get_frame(1)

    # We check opacity with one, two and three layers
    # Allow for a bit of tolerance (about 1%) to account
    # for rounding errors
    opacity1 = frame[50, 10]
    opacity2 = frame[50, 30]
    opacity3 = frame[50, 50]
    assert abs(opacity1 - 0.3) < 0.01
    assert abs(opacity2 - 0.51) < 0.01
    assert abs(opacity3 - 0.657) < 0.01


def test_compositing_with_transparency_colors(util):
    # Has one R 30%, one G 30%, one B 30%
    clip1 = ColorClip((100, 100), (255, 0, 0, 76.5)).with_duration(2)
    clip2 = ColorClip((50, 50), (0, 255, 0, 76.5)).with_duration(2)
    clip3 = ColorClip((25, 25), (0, 0, 255, 76.5)).with_duration(2)

    compostite_clip1 = CompositeVideoClip(
        [clip1, clip2.with_position(("center", "center"))]
    )
    compostite_clip2 = CompositeVideoClip(
        [compostite_clip1, clip3.with_position(("center", "center"))]
    )

    # Load output file and check transparency
    frame = compostite_clip2.get_frame(1)
    mask = compostite_clip2.mask.get_frame(1)

    # We check color with 1 layer
    # We add a bit of tolerance (about 1%) to account
    # For possible rounding errors
    color1 = frame[50, 10]
    opacity1 = mask[50, 10]
    assert np.allclose(color1, [255, 0, 0], rtol=0.01)
    assert abs(opacity1 - 0.3) < 0.01

    # With 2 layers
    color2 = frame[50, 30]
    opacity2 = mask[50, 30]
    assert np.allclose(color2, [105, 150, 0], rtol=0.01)
    assert abs(opacity2 - 0.51) < 0.01

    # With 3 layers
    color3 = frame[50, 50]
    opacity3 = mask[50, 50]
    assert np.allclose(color3, [57, 82, 116], rtol=0.01)
    assert abs(opacity3 - 0.657) < 0.01


def test_slide_in():
    duration = 0.1
    size = (10, 1)
    fps = 10
    color = (255, 0, 0)

    # left and right sides
    clip = ColorClip(
        color=color,
        duration=duration,
        size=size,
    ).with_fps(fps)

    for side in ["left", "right"]:
        new_clip = CompositeVideoClip(
            [clip.with_effects([vfx.SlideIn(duration, side)])]
        )

        for t in np.arange(0, duration, duration / fps):
            n_reds, n_reds_expected = (0, int(t * 100))

            if t:
                assert n_reds_expected

            if n_reds_expected == 7:  # skip 7 due to inaccurate frame
                continue

            for r, g, b in new_clip.get_frame(t)[0]:
                if r == color[0] and g == color[1] and g == color[2]:
                    n_reds += 1

            assert n_reds == n_reds_expected

    # top and bottom sides
    clip = ColorClip(
        color=color,
        duration=duration,
        size=(size[1], size[0]),
    ).with_fps(fps)

    for side in ["top", "bottom"]:
        new_clip = CompositeVideoClip(
            [clip.with_effects([vfx.SlideIn(duration, side)])]
        )
        for t in np.arange(0, duration, duration / fps):
            n_reds, n_reds_expected = (0, int(t * 100))

            if t:
                assert n_reds_expected

            if n_reds_expected == 7:  # skip 7 due to inaccurate frame
                continue

            for row in new_clip.get_frame(t):
                r, g, b = row[0]

                if r == color[0] and g == color[1] and g == color[2]:
                    n_reds += 1

            assert n_reds == n_reds_expected


def test_slide_out():
    duration = 0.1
    size = (11, 1)
    fps = 10
    color = (255, 0, 0)

    # left and right sides
    clip = ColorClip(
        color=color,
        duration=duration,
        size=size,
    ).with_fps(fps)

    for side in ["left", "right"]:
        new_clip = CompositeVideoClip(
            [clip.with_effects([vfx.SlideOut(duration, side)])]
        )

        for t in np.arange(0, duration, duration / fps):
            n_reds, n_reds_expected = (0, round(11 - t * 100, 6))

            if t:
                assert n_reds_expected

            for r, g, b in new_clip.get_frame(t)[0]:
                if r == color[0] and g == color[1] and g == color[2]:
                    n_reds += 1

            assert n_reds == n_reds_expected

    # top and bottom sides
    clip = ColorClip(
        color=color,
        duration=duration,
        size=(size[1], size[0]),
    ).with_fps(fps)

    for side in ["top", "bottom"]:
        new_clip = CompositeVideoClip(
            [clip.with_effects([vfx.SlideOut(duration, side)])]
        )
        for t in np.arange(0, duration, duration / fps):
            n_reds, n_reds_expected = (0, round(11 - t * 100, 6))

            if t:
                assert n_reds_expected

            for row in new_clip.get_frame(t):
                r, g, b = row[0]

                if r == color[0] and g == color[1] and g == color[2]:
                    n_reds += 1

            assert n_reds == n_reds_expected


def test_concatenate_with_masks(util):
    video_without_mask = (
        ColorClip(size=(10, 10), color=(255, 0, 0)).with_duration(1).with_fps(1)
    )
    video_with_mask = (
        ColorClip(size=(5, 5), color=(0, 255, 0))
        .with_duration(1)
        .with_fps(1)
        .with_mask()
    )

    output = os.path.join(util.TMP_DIR, "test_concatenate_with_masks.mp4")
    concatenate_videoclips([video_without_mask, video_with_mask]).write_videofile(
        output
    )


def reference_composite_rgb(background_rgba, overlays):
    """Slow, straightforward Pillow reference compositing of `overlays`
    over a full-size RGBA `background_rgba`, returning the RGB picture.

    This is the full-size alpha-compositing path that the optimized opaque
    background path must remain pixel-identical to.
    """
    background = np.array(background_rgba.convert("RGBA"))
    for overlay_rgba, pos in overlays:
        before = Image.fromarray(background)
        canvas = Image.new("RGBA", before.size, (0, 0, 0, 0))
        canvas.paste(overlay_rgba, pos)
        background = np.array(Image.alpha_composite(before, canvas))
    return background[:, :, :3].copy()


def reference_overlays():
    """RGBA images and positions matching the HUD overlays of `hud_clips`."""
    overlays = []
    for i in range(HUD_COUNT):
        array = np.full(HUD_SIZE[::-1] + (4,), 255, dtype=np.uint8)
        array[..., 3] = HUD_ALPHA
        pos = (
            (i * 37) % (HUD_GRID_SIZE[0] - HUD_SIZE[0]),
            (i * 53) % (HUD_GRID_SIZE[1] - HUD_SIZE[1]),
        )
        overlays.append((Image.fromarray(array), pos))
    return overlays


def test_hud_batch_opaque_background_matches_reference():
    """120 frames of 12 semi-transparent HUDs over an opaque background must
    stay pixel-identical to full alpha compositing, in contiguous memory.
    """
    clip = opaque_composite()
    overlays = reference_overlays()
    background = Image.new("RGBA", HUD_GRID_SIZE, (0, 0, 0, 255))
    expected = reference_composite_rgb(background, overlays)

    for i in range(HUD_FRAMES):
        frame = clip.get_frame(i / HUD_FPS)
        assert frame.shape == (HUD_GRID_SIZE[1], HUD_GRID_SIZE[0], 3)
        assert frame.dtype == np.uint8
        assert frame.flags["C_CONTIGUOUS"]
        assert int(frame.astype(np.int64).sum()) == HUD_EXPECTED_FRAME_PIXEL_SUM
        assert np.array_equal(frame, expected)


def test_hud_batch_opaque_bg_color_matches_reference():
    """Same as above when the opaque background is a ``bg_color``."""
    clip = CompositeVideoClip(hud_clips(), size=HUD_GRID_SIZE, bg_color=(0, 0, 0))
    overlays = reference_overlays()
    background = Image.new("RGBA", HUD_GRID_SIZE, (0, 0, 0, 255))
    expected = reference_composite_rgb(background, overlays)

    for i in range(HUD_FRAMES):
        frame = clip.get_frame(i / HUD_FPS)
        assert frame.flags["C_CONTIGUOUS"]
        assert int(frame.astype(np.int64).sum()) == HUD_EXPECTED_FRAME_PIXEL_SUM
        assert np.array_equal(frame, expected)


def test_hud_batch_transparent_background_stays_transparent():
    """120 frames over a transparent background keep their colors, masks and
    contiguity; the composite must not become opaque.
    """
    clip = transparent_composite()
    assert clip.mask is not None

    overlays = reference_overlays()
    background = Image.new("RGBA", HUD_GRID_SIZE, (0, 0, 0, 0))
    expected = reference_composite_rgb(background, overlays)

    for i in range(HUD_FRAMES):
        t = i / HUD_FPS
        frame = clip.get_frame(t)
        mask = clip.mask.get_frame(t)

        assert frame.shape == (HUD_GRID_SIZE[1], HUD_GRID_SIZE[0], 3)
        assert frame.dtype == np.uint8
        assert frame.flags["C_CONTIGUOUS"]
        assert mask.shape == (HUD_GRID_SIZE[1], HUD_GRID_SIZE[0])
        assert np.array_equal(frame, expected)

    # Transparency is preserved: an uncovered corner stays fully transparent
    # while a pixel under a single overlay is only partially opaque.
    mask = clip.mask.get_frame(0)
    assert mask[-1, -1] == 0
    assert 0 < mask[HUD_SIZE[1] // 2, HUD_SIZE[0] // 2] < 1


def test_hud_batch_export(util):
    """Both background kinds export through the regular video writer."""
    opaque_file = os.path.join(util.TMP_DIR, "test_hud_batch_opaque.mp4")
    transparent_file = os.path.join(util.TMP_DIR, "test_hud_batch_transparent.mp4")

    for clip, path in (
        (opaque_composite(), opaque_file),
        (transparent_composite(), transparent_file),
    ):
        clip.with_duration(HUD_FRAMES / HUD_FPS).with_fps(HUD_FPS).write_videofile(
            path, codec="libx264", preset="ultrafast", logger=None
        )
        assert os.path.isfile(path)


if __name__ == "__main__":
    pytest.main()
