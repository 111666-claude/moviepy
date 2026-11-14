"""Compositing tests for use with pytest."""

import os

import numpy as np

import pytest

from moviepy import *


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


# def test_blit_with_opacity():
#     # bitmap.mp4 has one second R, one second G, one second B
#     clip1 = VideoFileClip("media/bitmap.mp4")
#     # overlay same clip, shifted by 1 second, at half opacity
#     clip2 = (
#         VideoFileClip("media/bitmap.mp4")
#         .subclip(1, 2)
#         .with_start(0)
#         .with_end(2)
#         .with_opacity(0.5)
#     )
#     composite = CompositeVideoClip([clip1, clip2])
#     bt = ClipPixelTest(composite)

#     bt.expect_color_at(0.5, (0x7F, 0x7F, 0x00))
#     bt.expect_color_at(1.5, (0x00, 0x7F, 0x7F))
#     bt.expect_color_at(2.5, (0x00, 0x00, 0xFF))


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


def test_compositing_masks(util):
    """Mask composition must accumulate transparency multiplicatively."""
    # Has one R 30%, one G 30%, one B 30%
    clip1 = ColorClip((100, 100), (255, 0, 0, 76.5)).with_duration(2)
    clip2 = ColorClip((50, 50), (0, 255, 0, 76.5)).with_duration(2)
    clip3 = ColorClip((25, 25), (0, 0, 255, 76.5)).with_duration(2)

    compostite_clip1 = CompositeVideoClip(
        [clip1, clip2.with_position(("center", "center"))],
        bg_color=None,
    )
    compostite_clip2 = CompositeVideoClip(
        [compostite_clip1, clip3.with_position(("center", "center"))],
        bg_color=None,
    )

    frame = compostite_clip2.mask.get_frame(1)

    # We check opacity with one, two and three layers
    # Allow for a bit of tolerance (about 1%) to account for rounding errors
    opacity1 = frame[50, 10]
    opacity2 = frame[50, 30]
    opacity3 = frame[50, 50]
    assert abs(opacity1 - 0.3) < 0.01
    assert abs(opacity2 - 0.51) < 0.01
    assert abs(opacity3 - 0.657) < 0.01


def test_compositing_with_transparency_colors(util):
    """Colors and masks must both be preserved through nested compositions."""
    # Has one R 30%, one G 30%, one B 30%
    clip1 = ColorClip((100, 100), (255, 0, 0, 76.5)).with_duration(2)
    clip2 = ColorClip((50, 50), (0, 255, 0, 76.5)).with_duration(2)
    clip3 = ColorClip((25, 25), (0, 0, 255, 76.5)).with_duration(2)

    compostite_clip1 = CompositeVideoClip(
        [clip1, clip2.with_position(("center", "center"))],
        bg_color=None,  # No background color, so it should be transparent
    )
    compostite_clip2 = CompositeVideoClip(
        [compostite_clip1, clip3.with_position(("center", "center"))],
        bg_color=None,
    )

    frame = compostite_clip2.get_frame(1)
    mask = compostite_clip2.mask.get_frame(1)

    # We check color with 1 layer
    # We add a bit of tolerance (about 1%) to account for rounding errors
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
    assert np.allclose(color3, [57, 82, 116], atol=1)
    assert abs(opacity3 - 0.657) < 0.01


def test_compositing_nested_transparency(util):
    """A transparent composite nested in another one keeps its opacity.

    Two nested composites must not pre-multiply their colors against the
    transparent background: the flat colors and the accumulated mask must be
    preserved, and the opaque background must stay fully opaque.
    """
    clip1 = ColorClip((200, 200), (255, 0, 0)).with_duration(3)
    clip2 = ColorClip((100, 100), (0, 255, 0, 76.5)).with_duration(3)
    clip3 = ColorClip((50, 50), (0, 0, 255, 76.5)).with_duration(3)

    inner = CompositeVideoClip(
        [clip1, clip2.with_position(("center", "center"))],
        bg_color=None,
    )
    outer = CompositeVideoClip(
        [inner, clip3.with_position(("center", "center"))],
        bg_color=None,
    )

    frame = outer.get_frame(2)
    mask = outer.mask.get_frame(2)

    # The opaque red background keeps the whole composite opaque.
    assert np.array_equal(mask[100, 10], 1.0)

    assert np.allclose(frame[100, 10], [255, 0, 0], rtol=0.01)
    assert np.allclose(frame[100, 60], [179, 76, 0], rtol=0.01)
    assert np.allclose(frame[100, 100], [126, 53, 76], rtol=0.01)


def test_compositing_many_transparent_layers():
    """Many stacked semi-transparent layers must not swallow each other."""
    clip = ColorClip((128, 128), (10, 20, 30, 25.5)).with_duration(1)
    composite = CompositeVideoClip(
        [clip.with_position("center") for _ in range(32)],
        size=(128, 128),
        bg_color=None,
    )

    mask = composite.mask.get_frame(0.5)
    assert abs(mask[64, 64] - 0.965663) < 1e-4


def test_compositing_transparent_moving_position():
    """Mask and color composition must follow a moving clip position."""
    clip1 = ColorClip((200, 200), (255, 0, 0)).with_duration(3)
    clip2 = ColorClip((100, 100), (0, 255, 0, 76.5)).with_duration(3)
    clip3 = ColorClip((50, 50), (0, 0, 255, 76.5)).with_duration(3)

    inner = CompositeVideoClip(
        [clip2.with_position("center"), clip3.with_position("center")],
        size=(100, 100),
        bg_color=None,
    )
    position = lambda t: (int(50 + t * 20), int(50 + t * 20))
    composite = CompositeVideoClip(
        [clip1, inner.with_position(position)],
        size=(200, 200),
        bg_color=None,
    )

    frame = composite.get_frame(2)
    mask = composite.mask.get_frame(2)

    # At t=2 the inner composite starts at (90, 90).
    assert np.allclose(frame[60, 60], [255, 0, 0], rtol=0.01)
    assert np.allclose(frame[100, 100], [179, 76, 0], rtol=0.01)
    assert np.allclose(frame[140, 140], [125, 54, 76], atol=1)
    assert np.array_equal(mask[90:150, 90:150], np.ones((60, 60)))


def test_compositing_transparent_crop_consistency():
    """Cropping a transparent composite keeps frame and mask aligned."""
    clip1 = ColorClip((100, 100), (255, 0, 0, 76.5)).with_duration(2)
    clip2 = ColorClip((50, 50), (0, 255, 0, 76.5)).with_duration(2)

    composite = CompositeVideoClip(
        [clip1, clip2.with_position("center")], bg_color=None
    ).with_effects([vfx.Crop(x1=10, y1=10, x2=90, y2=90)])

    frame = composite.get_frame(1)
    mask = composite.mask.get_frame(1)

    assert frame.shape == (80, 80, 3)
    assert mask.shape == (80, 80)
    assert abs(mask[0, 0] - 0.3) < 0.01
    assert abs(mask[40, 40] - 0.51) < 0.01
    assert np.allclose(frame[40, 40], [105, 150, 0], rtol=0.01)


def test_compositing_transparent_export_consistency(util):
    """Exported nested/moving/cropped frames must match the composite."""
    clip1 = ColorClip((200, 200), (255, 0, 0)).with_duration(2)
    clip2 = ColorClip((100, 100), (0, 255, 0, 76.5)).with_duration(2)
    clip3 = ColorClip((50, 50), (0, 0, 255, 76.5)).with_duration(2)

    inner = CompositeVideoClip(
        [clip2.with_position("center"), clip3.with_position("center")],
        size=(100, 100),
        bg_color=None,
    )
    composite = CompositeVideoClip(
        [clip1, inner.with_position(lambda t: (int(50 + t * 10), int(50 + t * 10)))],
        size=(200, 200),
        bg_color=None,
    ).with_fps(5)

    for name, clip in (
        ("nested", composite),
        ("cropped", composite.with_effects([vfx.Crop(x1=40, y1=40, x2=160, y2=160)])),
    ):
        filename = os.path.join(util.TMP_DIR, f"transparent_{name}.mov")
        if os.path.isfile(filename):
            os.remove(filename)

        clip.write_videofile(filename, codec="prores", fps=5, logger=None, audio=False)

        exported = VideoFileClip(filename, has_mask=True)
        try:
            for t in (0.2, 0.6):
                expected_frame = clip.get_frame(t)
                expected_mask = clip.mask.get_frame(t)
                frame = exported.get_frame(t)
                mask = exported.mask.get_frame(t)

                assert frame.shape == expected_frame.shape
                assert mask.shape == expected_mask.shape
                # ProRes is lossy, so allow one level of rounding per channel.
                assert np.abs(frame.astype(int) - expected_frame.astype(int)).max() <= 1
                assert np.abs(mask - expected_mask).max() <= 0.01
        finally:
            exported.close()


if __name__ == "__main__":
    pytest.main()
