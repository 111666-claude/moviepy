"""Tests for the ``vfx.Deflicker`` temporal deflickering effect."""

import os

import numpy as np

import pytest

from moviepy import *
from moviepy.video.fx.Deflicker import Deflicker, _BrightnessCache


def make_uniform_clip(levels, fps=10, size=(8, 8)):
    """Build a clip whose frame i is a uniform RGB image of ``levels[i]``."""
    frames = np.array(
        [np.full((size[1], size[0], 3), level, dtype=np.uint8) for level in levels]
    )
    duration = len(levels) / fps

    def frame_function(t):
        index = min(int(t * fps), len(frames) - 1)
        return frames[index]

    return VideoClip(frame_function, duration=duration).with_fps(fps)


def mean_brightness(frame):
    return float(np.mean(frame))


def test_deflicker_reduces_alternating_flicker():
    # Exposure jumps around a base level, like a real time-lapse shot in
    # auto-exposure mode: consecutive frames vary irregularly rather than
    # following a perfectly alternating two-level pattern.
    rng = np.random.default_rng(2)
    levels = np.clip(150 + rng.normal(0, 45, 32), 40, 230).round().astype(int)
    clip = make_uniform_clip(levels, fps=10)

    result = clip.with_effects([Deflicker(radius=3, max_gain=0.5)])
    output_levels = np.array(
        [mean_brightness(result.get_frame(i / 10)) for i in range(len(levels))]
    )

    # Interior frames have a full window, so their brightness spread is
    # compressed to at least ~1/3 below the input spread (the max_gain cap
    # deliberately prevents a stronger, unsafe correction).
    interior = slice(4, -4)
    assert np.std(output_levels[interior]) < np.std(levels[interior]) * 2 / 3
    assert np.ptp(output_levels[interior]) < np.ptp(levels[interior])

    # Every individual frame moves toward (or reaches) the median of the
    # actual (possibly boundary-clipped) window used by the effect.
    for i, level in enumerate(levels):
        start, end = result.deflicker_window_bounds(i)
        target = np.median(levels[start : end + 1])
        assert abs(output_levels[i] - target) <= abs(level - target) + 1


def test_deflicker_respects_max_gain():
    levels = [200, 100] * 5
    clip = make_uniform_clip(levels, fps=10)

    result = clip.with_effects([Deflicker(radius=1, max_gain=0.1)])

    for i, level in enumerate(levels):
        output = mean_brightness(result.get_frame(i / 10))
        # The correction cannot exceed +/- 10% of the original brightness.
        assert output <= level * 1.1 + 1
        assert output >= level * 0.9 - 1
        # And it must move toward the median, not away from it.
        assert abs(output - 150) <= abs(level - 150) + 1


def test_deflicker_preserves_shot_levels_across_scene_cut():
    # Two genuinely different shots joined together.
    levels = [160] * 8 + [40] * 8
    clip = make_uniform_clip(levels, fps=10)

    result = clip.with_effects(
        [Deflicker(radius=3, max_gain=0.5, scene_cut_threshold=0.5)]
    )

    # The boundary between frames 7 and 8 is a 75% relative brightness jump,
    # well above the 0.5 threshold: windows must be contracted at the cut so
    # the two shots never mix.
    for center in range(len(levels)):
        start, end = result.deflicker_window_bounds(center)
        if center <= 7:
            assert end <= 7  # no dark frame enters a bright-shot window
        else:
            assert start >= 8  # no bright frame enters a dark-shot window

    # Each shot is already internally stable, so with the cut isolating the
    # windows every frame is left untouched.
    output_levels = [
        mean_brightness(result.get_frame(i / 10)) for i in range(len(levels))
    ]
    for i, level in enumerate(levels):
        assert output_levels[i] == pytest.approx(level, abs=1)


def test_deflicker_without_cut_detection_blends_the_shots():
    # Disabling scene cut detection lets the median window cross the cut.
    levels = [160] * 8 + [40] * 8
    clip = make_uniform_clip(levels, fps=10)

    result = clip.with_effects(
        [Deflicker(radius=3, max_gain=0.5, scene_cut_threshold=None)]
    )

    # With cut detection off, frames next to the cut see the other shot in
    # their window.
    start_at_7, _ = result.deflicker_window_bounds(7)
    _, end_at_8 = result.deflicker_window_bounds(8)
    assert start_at_7 == 4
    assert end_at_8 == 11

    # And with cut detection on, the same windows stop at the boundary.
    result_cuts = clip.with_effects(
        [Deflicker(radius=3, max_gain=0.5, scene_cut_threshold=0.5)]
    )
    assert result_cuts.deflicker_window_bounds(7) == (4, 7)
    assert result_cuts.deflicker_window_bounds(8) == (8, 11)


def test_deflicker_preserves_duration_fps_audio_and_mask():
    levels = [200, 100] * 6
    clip = make_uniform_clip(levels, fps=12)
    clip = clip.with_duration(1.0)

    audio = AudioClip(lambda t: np.sin(440 * 2 * np.pi * t), duration=1.0, fps=44100)
    mask = ColorClip((8, 8), color=0.5, is_mask=True, duration=1.0)
    clip = clip.with_audio(audio).with_mask(mask)

    result = clip.with_effects([Deflicker(radius=2)])

    assert result.duration == clip.duration
    assert result.fps == clip.fps
    assert result.n_frames == clip.n_frames
    assert result.size == clip.size
    assert result.audio is not None
    assert result.audio.duration == audio.duration
    assert result.mask is not None
    # The mask is passed through untouched.
    assert np.array_equal(result.mask.get_frame(0.3), clip.mask.get_frame(0.3))


def test_deflicker_random_and_sequential_access_match():
    rng = np.random.default_rng(42)
    levels = rng.integers(60, 220, size=24)
    clip = make_uniform_clip(levels, fps=8)

    result = clip.with_effects([Deflicker(radius=3, max_gain=0.3)])

    random_order = [7, 0, 15, 3, 20, 11, 23, 1, 18, 12]
    random_frames = [result.get_frame(i / 8).copy() for i in random_order]

    # Sequential render of the whole clip.
    sequential = list(result.iter_frames())

    for frame_index, frame in zip(random_order, random_frames):
        assert np.array_equal(frame, sequential[frame_index])


def test_deflicker_cache_is_bounded():
    levels = [200, 100] * 20
    clip = make_uniform_clip(levels, fps=10)
    effect = Deflicker(radius=2)
    result = clip.with_effects([effect])

    # Walk the whole clip sequentially.
    for i in range(len(levels)):
        result.get_frame(i / 10)

    cache_obj = result.deflicker_cache
    assert isinstance(cache_obj, _BrightnessCache)
    assert len(cache_obj._values) <= cache_obj.maxsize
    assert cache_obj.maxsize == 2 * effect.radius + 3


def test_deflicker_single_frame_clip():
    clip = make_uniform_clip([123], fps=10)
    result = clip.with_effects([Deflicker(radius=5)])

    # The only available frame is its own median: it must stay unchanged.
    assert mean_brightness(result.get_frame(0)) == pytest.approx(123, abs=1)
    assert result.n_frames == 1


def test_deflicker_clipped_window_at_clip_boundaries():
    levels = [180, 90, 150, 170, 120]
    clip = make_uniform_clip(levels, fps=10)
    result = clip.with_effects([Deflicker(radius=10, max_gain=0.9)])

    # No index error at the head and tail of the clip, and every output
    # stays in the valid uint8 range.
    for i in range(len(levels)):
        frame = result.get_frame(i / 10)
        assert frame.min() >= 0
        assert frame.max() <= 255


def test_deflicker_zero_brightness_frames_are_untouched():
    levels = [0, 0, 0, 0, 0]
    clip = make_uniform_clip(levels, fps=10)
    result = clip.with_effects([Deflicker(radius=2, max_gain=0.5)])

    for i in range(len(levels)):
        assert np.all(result.get_frame(i / 10) == 0)

    # A single dark frame between bright frames cannot be brightened
    # (dividing by zero brightness is undefined): it stays black.
    levels = [200, 200, 0, 200, 200]
    clip = make_uniform_clip(levels, fps=10)
    result = clip.with_effects([Deflicker(radius=2, max_gain=0.5)])
    assert np.all(result.get_frame(2 / 10) == 0)


def test_deflicker_supports_variable_size_frames():
    fps = 4
    small = np.full((4, 4, 3), 100, dtype=np.uint8)
    large = np.full((8, 8, 3), 200, dtype=np.uint8)
    frames = [small, large, small, large]

    clip = VideoClip(
        lambda t: frames[min(int(t * fps), 3)],
        duration=1.0,
        has_constant_size=False,
    ).with_fps(fps)

    result = clip.with_effects([Deflicker(radius=1, max_gain=0.5)])

    # Output frames keep each input frame's shape and stay valid.
    for i, frame in enumerate(frames):
        out = result.get_frame(i / fps)
        assert out.shape == frame.shape
        assert out.min() >= 0 and out.max() <= 255


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"radius": 0}, "'radius' must be a positive integer"),
        ({"radius": -2}, "'radius' must be a positive integer"),
        ({"radius": 1.5}, "'radius' must be a positive integer"),
        ({"radius": True}, "'radius' must be a positive integer"),
        ({"radius": 1, "max_gain": 0}, None),  # valid: correction disabled
        ({"radius": 1, "max_gain": 1}, "0 <= max_gain < 1"),
        ({"radius": 1, "max_gain": -0.1}, "0 <= max_gain < 1"),
        (
            {"radius": 1, "scene_cut_threshold": 1.2},
            "'scene_cut_threshold' must be between 0 and 1 inclusive",
        ),
        (
            {"radius": 1, "scene_cut_threshold": -0.1},
            "'scene_cut_threshold' must be between 0 and 1 inclusive",
        ),
    ],
)
def test_deflicker_validates_parameters(kwargs, match):
    clip = make_uniform_clip([100, 120], fps=10)
    if match is None:
        clip.with_effects([Deflicker(**kwargs)])  # must not raise
    else:
        with pytest.raises((ValueError, TypeError), match=match):
            clip.with_effects([Deflicker(**kwargs)])


def test_deflicker_requires_fps():
    clip = VideoClip(lambda t: np.zeros((4, 4, 3), dtype=np.uint8), duration=1.0)
    assert getattr(clip, "fps", None) is None
    with pytest.raises(ValueError, match="'fps' attribute"):
        clip.with_effects([Deflicker(radius=1)])


def test_deflicker_write_videofile_matches_random_access(util):
    levels = [210, 110] * 8
    clip = make_uniform_clip(levels, fps=10)
    result = clip.with_effects([Deflicker(radius=2, max_gain=0.4)])

    filename = os.path.join(util.TMP_DIR, "deflicker_export.mp4")
    result.write_videofile(filename, fps=10, logger=None)

    reloaded = VideoFileClip(filename)
    try:
        # Same number of frames and the same corrected brightness at every
        # frame index, so sequential export and random access agree.
        assert reloaded.n_frames == result.n_frames
        for i in range(result.n_frames):
            exported = mean_brightness(reloaded.get_frame(i / 10))
            in_memory = mean_brightness(result.get_frame(i / 10))
            assert exported == pytest.approx(in_memory, abs=2)
    finally:
        reloaded.close()
        os.remove(filename)


def test_deflicker_synthetic_clip_luminance_and_frame_count():
    """End-to-end check on the alternating-brightness synthetic clip.

    The corrected clip must keep exactly the same frame count/duration while
    the frame-to-frame luminance spread is strongly reduced.
    """
    rng = np.random.default_rng(3)
    levels = np.clip(150 + rng.normal(0, 45, 32), 40, 230).round().astype(int)
    clip = make_uniform_clip(levels, fps=24)
    result = clip.with_effects([Deflicker(radius=5, max_gain=0.5)])

    assert result.n_frames == len(levels)
    assert result.duration == clip.duration

    output_levels = np.array(
        [mean_brightness(result.get_frame(i / 24)) for i in range(len(levels))]
    )
    interior = slice(6, -6)
    assert np.std(output_levels[interior]) < np.std(levels[interior]) * 2 / 3
    assert np.ptp(output_levels[interior]) < np.ptp(levels[interior])
