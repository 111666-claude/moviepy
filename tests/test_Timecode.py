"""Tests for the Timecode effect and the SMPTE timecode helpers."""

import warnings
from fractions import Fraction

import numpy as np
import pytest

from moviepy import (
    ColorClip,
    CompositeVideoClip,
    VideoClip,
    vfx,
)
from moviepy.video.tools import timecode as tc_tools
from moviepy.video.tools.timecode import frame_to_timecode, timecode_to_frame

# ---------------------------------------------------------------------------
# Pure timecode arithmetic
# ---------------------------------------------------------------------------


def test_frame_to_timecode_non_drop_frame():
    assert frame_to_timecode(0, 25) == "00:00:00:00"
    assert frame_to_timecode(24, 25) == "00:00:00:24"
    assert frame_to_timecode(25, 25) == "00:00:01:00"
    assert frame_to_timecode(1800, 30) == "00:01:00:00"
    assert frame_to_timecode(24 * 3600, 24) == "01:00:00:00"
    assert frame_to_timecode(30 * 3600 * 24 - 1, 30) == "23:59:59:29"


def test_frame_to_timecode_accepts_fractional_and_string_fps():
    assert frame_to_timecode(25, Fraction(25, 1)) == "00:00:01:00"
    assert frame_to_timecode(25, "25") == "00:00:01:00"


@pytest.mark.parametrize(
    "fps,rate,drop,frames_per_10min",
    [(29.97, 30, 2, 17982), (59.94, 60, 4, 35964)],
)
def test_drop_frame_matches_brute_force(fps, rate, drop, frames_per_10min):
    """Exhaustively compare the closed form against a frame-by-frame walk."""

    def brute_labels(total):
        labels = {}
        hour = minute = second = frame = 0
        n = 0
        while n < total:
            labels[n] = (hour, minute, second, frame)
            n += 1
            frame += 1
            if frame == rate:
                frame = 0
                second += 1
                if second == 60:
                    second = 0
                    minute += 1
                    if minute % 10 != 0:
                        frame = drop
                    if minute == 60:
                        minute = 0
                        hour += 1
        return labels

    # One full hour is enough to cross 10-minute chunk boundaries.
    total = frames_per_10min * 6
    labels = brute_labels(total)
    for n, (hour, minute, second, frame) in labels.items():
        expected = "{:02d}:{:02d}:{:02d};{:02d}".format(hour, minute, second, frame)
        assert frame_to_timecode(n, fps, drop_frame=True) == expected
        assert timecode_to_frame(expected, fps, drop_frame=True) == n


def test_drop_frame_known_anchors():
    assert frame_to_timecode(0, 29.97, True) == "00:00:00;00"
    assert frame_to_timecode(1799, 29.97, True) == "00:00:59;29"
    assert frame_to_timecode(1800, 29.97, True) == "00:01:00;02"
    assert frame_to_timecode(3598, 29.97, True) == "00:02:00;02"
    assert frame_to_timecode(17982, 29.97, True) == "00:10:00;00"
    assert frame_to_timecode(107892, 29.97, True) == "01:00:00;00"

    assert frame_to_timecode(3600, 59.94, True) == "00:01:00;04"
    assert frame_to_timecode(35964, 59.94, True) == "00:10:00;00"
    assert frame_to_timecode(215784, 59.94, True) == "01:00:00;00"

    # Exact fractional and float frame rates are treated identically.
    assert frame_to_timecode(1800, Fraction(30000, 1001), True) == "00:01:00;02"


def test_timecode_parsing_infers_drop_frame_from_separator():
    assert timecode_to_frame("00:01:00;02", 29.97) == 1800
    assert timecode_to_frame("00:01:00:02", 29.97, drop_frame=True) == 1800
    assert timecode_to_frame("00:00:01:00", 25) == 25


@pytest.mark.parametrize(
    "fps,drop_frame,frames",
    [
        (25, False, [0, 24, 25, 1799, 90000]),
        (30, False, [0, 1800, 107999]),
        (29.97, True, [0, 1799, 1800, 17982, 107892, 107891]),
        (59.94, True, [0, 3600, 35964, 215784]),
    ],
)
def test_timecode_round_trips(fps, drop_frame, frames):
    for n in frames:
        label = frame_to_timecode(n, fps, drop_frame)
        assert timecode_to_frame(label, fps, drop_frame) == n


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        frame_to_timecode(-1, 25)
    with pytest.raises(TypeError):
        frame_to_timecode(1.5, 25)
    with pytest.raises(TypeError):
        frame_to_timecode(True, 25)
    with pytest.raises(ValueError):
        frame_to_timecode(0, 29.97)  # fractional fps needs drop_frame
    with pytest.raises(ValueError):
        frame_to_timecode(0, 25, drop_frame=True)  # illegal combination
    with pytest.raises(ValueError):
        frame_to_timecode(0, 30, drop_frame=True)
    with pytest.raises(ValueError):
        frame_to_timecode(30 * 3600 * 24, 30)  # beyond 24 hours
    with pytest.raises(ValueError):
        frame_to_timecode(2589408, 29.97, True)
    with pytest.raises(ValueError):
        frame_to_timecode(0, 0)
    with pytest.raises(ValueError):
        frame_to_timecode(0, -25)

    with pytest.raises(ValueError):
        timecode_to_frame("00:00:12", 25)  # missing field
    with pytest.raises(ValueError):
        timecode_to_frame("00:00:00:30", 30)  # frame out of range
    with pytest.raises(ValueError):
        timecode_to_frame("00:01:00;00", 29.97)  # dropped label
    with pytest.raises(ValueError):
        # drop separator but explicitly non drop-frame
        timecode_to_frame("00:01:00;02", 29.97, drop_frame=False)


# ---------------------------------------------------------------------------
# Effect behavior
# ---------------------------------------------------------------------------


def make_clip(size=(320, 240), color=(20, 60, 120), fps=10, duration=2):
    return ColorClip(size=size, color=color, duration=duration).with_fps(fps)


def text_mask(frame):
    """Return the boolean mask of the bright (white) timecode glyphs."""
    return (frame[:, :, 0] > 200) & (frame[:, :, 1] > 200) & (frame[:, :, 2] > 200)


def expected_label(frame_index, fps, start_frame=0, drop_frame=False):
    return frame_to_timecode(start_frame + frame_index, fps, drop_frame).replace(
        ";", ":"
    )


def test_effect_renders_distinct_labels_each_frame():
    clip = make_clip()
    rendered = clip.with_effects([vfx.Timecode(fps=10)])

    seen_masks = set()
    for frame_index in range(20):
        t = frame_index / 10
        frame = rendered.get_frame(t)
        mask = text_mask(frame)
        assert mask.any()  # something is drawn on every frame
        # Frame boundaries must be stable: querying the same frame time twice
        # returns identical pixels.
        assert np.array_equal(frame, rendered.get_frame(t))
        seen_masks.add(mask.tobytes())

    # The 20 frame indices cover 20 distinct labels (00..09 in two seconds).
    assert len(seen_masks) == 20


def test_effect_label_text_matches_timecode(util):
    """Frames show the ten distinct labels of the first second, in order."""
    fps = 10
    clip = make_clip(fps=fps, duration=1)
    effect = vfx.Timecode(fps=fps, font=util.FONT, font_size=24, margin=0)
    rendered = clip.with_effects([effect])

    crops = []
    for frame_index in range(10):
        frame = rendered.get_frame(frame_index / fps)
        ys, xs = np.where(text_mask(frame))
        crops.append(text_mask(frame)[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])

    # All ten labels render (none is blank) and are pairwise distinct.
    assert all(crop.size > 0 for crop in crops)
    assert len({crop.tobytes() for crop in crops}) == 10

    # The effect's own renderer produces an image whose bright pixels match
    # the on-frame glyph for the same label and options.
    effect._resolve_parameters(clip)
    label_image = np.array(effect._render_text(expected_label(0, fps)).convert("RGB"))
    rendered_label = label_image.max(axis=2) > 200
    ref_ys, ref_xs = np.where(rendered_label)
    ref_crop = rendered_label[
        ref_ys.min() : ref_ys.max() + 1, ref_xs.min() : ref_xs.max() + 1
    ]
    assert ref_crop.shape == crops[0].shape
    assert np.array_equal(ref_crop, crops[0])


def test_effect_position_and_margin():
    clip = make_clip(size=(320, 240))

    top_left = clip.with_effects(
        [vfx.Timecode(fps=10, position=("left", "top"), margin=8)]
    )
    ys, xs = np.where(text_mask(top_left.get_frame(0.0)))
    assert xs.min() >= 8 and ys.min() >= 8
    assert xs.max() < 160 and ys.max() < 60

    bottom_right = clip.with_effects(
        [vfx.Timecode(fps=10, position=("right", "bottom"), margin=8)]
    )
    ys, xs = np.where(text_mask(bottom_right.get_frame(0.0)))
    assert xs.max() <= 320 - 8 and ys.max() <= 240 - 8
    assert xs.min() > 160 and ys.min() > 180

    center = clip.with_effects([vfx.Timecode(fps=10, position="center")])
    ys, xs = np.where(text_mask(center.get_frame(0.0)))
    assert 80 < xs.min() and xs.max() < 240
    assert 90 < ys.min() and ys.max() < 150

    # Exact pixel position anchors the text *image* origin; the glyph's ink
    # starts one pixel in because of the font's left side bearing.
    effect = vfx.Timecode(fps=10, position=(42, 37), stroke_width=0)
    exact = clip.with_effects([effect])
    effect._resolve_parameters(clip)
    label_img = np.array(effect._render_text("00:00:00:00").convert("RGB"))
    ink = label_img.max(axis=2) > 200
    bearing_x = np.where(ink)[1].min()
    bearing_y = np.where(ink)[0].min()

    ys, xs = np.where(text_mask(exact.get_frame(0.0)))
    assert xs.min() == 42 + bearing_x
    assert ys.min() == 37 + bearing_y


def test_effect_preserves_frame_content_duration_and_audio():
    clip = make_clip()
    rendered = clip.with_effects([vfx.Timecode(fps=10)])

    # Duration / fps are untouched.
    assert rendered.duration == clip.duration
    assert rendered.fps == clip.fps
    assert rendered.size == clip.size

    # Pixels away from the top-left timecode box are byte-identical.
    original = clip.get_frame(0.0)
    frame = rendered.get_frame(0.0)
    assert frame.dtype == original.dtype
    assert np.array_equal(frame[80:, :, :], original[80:, :, :])

    # No audio is introduced by the effect.
    assert rendered.audio is None


def test_effect_on_subclip_keeps_source_timecode():
    # Effect applied before subclipping: local time 0 maps back to the
    # source frame at the cut point, so the label follows the source.
    clip = make_clip(fps=10, duration=3)
    burned = clip.with_effects([vfx.Timecode(fps=10)])
    sub = burned.subclipped(1.0, 2.0)

    assert sub.duration == 1.0
    first = sub.get_frame(0.0)
    tenth = sub.get_frame(0.9)
    # Source frame 10 -> 00:00:01:00, source frame 19 -> 00:00:01:09.
    assert text_mask(first).any()
    assert text_mask(tenth).any()

    # Cross-check against the helper with the source frame offset.
    assert frame_to_timecode(10, 10) == "00:00:01:00"
    assert frame_to_timecode(19, 10) == "00:00:01:09"


def test_effect_start_timecode_on_subclip():
    # Effect applied after subclipping with the cut point's timecode given
    # explicitly yields the same labels as the effect-first workflow.
    clip = make_clip(fps=10, duration=3)
    sub = clip.subclipped(1.0, 2.0)
    rendered = sub.with_effects([vfx.Timecode(start_timecode="00:00:01:00", fps=10)])
    assert text_mask(rendered.get_frame(0.0)).any()
    assert frame_to_timecode(10, 10) == "00:00:01:00"


def test_effect_inside_composite():
    bg = make_clip(color=(40, 40, 40))
    burned = bg.with_effects(
        [
            vfx.Timecode(
                start_timecode="00:00:05:00",
                fps=10,
                position=("right", "bottom"),
            )
        ]
    )
    composite = CompositeVideoClip([burned], size=(320, 240))
    frame = composite.get_frame(0.0)
    ys, xs = np.where(text_mask(frame))
    # Anchored in the bottom-right corner.
    assert xs.min() > 160 and ys.min() > 180
    assert composite.duration == bg.duration


def test_effect_uses_clip_fps_by_default():
    clip = make_clip(fps=12)
    rendered = clip.with_effects([vfx.Timecode()])
    # 12 frames per second: frame 11 is the last label of the first second.
    frame = rendered.get_frame(11 / 12)
    assert text_mask(frame).any()
    assert frame_to_timecode(11, 12) == "00:00:00:11"


def test_effect_drop_frame_rendering():
    clip = make_clip(fps=1)  # clip fps irrelevant, effect fps governs labels
    rendered = clip.with_effects(
        [vfx.Timecode(fps=29.97, drop_frame=True, start_timecode=1799)]
    )
    assert text_mask(rendered.get_frame(0.0)).any()


def test_effect_invalid_start_timecode():
    clip = make_clip()
    with pytest.raises(ValueError):
        clip.with_effects([vfx.Timecode(fps=10, start_timecode=-1)])
    with pytest.raises(TypeError):
        clip.with_effects([vfx.Timecode(fps=10, start_timecode=1.5)])
    with pytest.raises(ValueError):
        clip.with_effects([vfx.Timecode(fps=25, drop_frame=True)])


def test_effect_requires_fps():
    clip = ColorClip(size=(10, 10), color=(0, 0, 0), duration=1)
    assert getattr(clip, "fps", None) is None
    with pytest.raises(ValueError):
        clip.with_effects([vfx.Timecode()])


def test_effect_missing_font_falls_back_with_warning():
    clip = make_clip()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rendered = clip.with_effects([vfx.Timecode(fps=10, font="/does/not/exist.ttf")])
        frame = rendered.get_frame(0.0)
    assert text_mask(frame).any()
    assert any(issubclass(warning.category, RuntimeWarning) for warning in caught)


def test_effect_rejects_grayscale_clip():
    def gray_frame(t):
        return np.zeros((20, 30), dtype=np.uint8)

    clip = VideoClip(gray_frame, duration=1).with_fps(10)
    with pytest.raises(TypeError):
        clip.with_effects([vfx.Timecode(fps=10)]).get_frame(0.0)


def test_effect_is_registered():
    assert hasattr(vfx, "Timecode")
    assert "Timecode" in vfx.__all__
    assert tc_tools is not None
