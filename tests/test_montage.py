"""Tests for the audio-onset montage assembly entry point.

The montage is checked end to end with fully synthetic material:

- a pulse audio track whose onset times are known exactly,
- solid color clips whose frames are trivially identifiable.

This keeps the tests fast and deterministic, and avoids any ffmpeg dependency.
"""

import numpy as np

import pytest

from moviepy import AudioClip, ColorClip
from moviepy.audio.tools.cuts import find_audio_onsets, find_audio_period
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
from moviepy.video.tools.cuts import montage_on_onsets


AUDIO_FPS = 1000
VIDEO_FPS = 10
SIZE = (32, 24)
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def pulse_audio(
    pulse_times,
    duration=2.0,
    fps=AUDIO_FPS,
    pulse_width=0.04,
    noise=0.0,
    nchannels=1,
    seed=0,
):
    """Builds a synthetic audio clip made of short unit-energy pulses.

    ``pulse_times`` are the exact onset times the detector is expected to find.
    """
    rng = np.random.default_rng(seed)

    def frame_function(t):
        t = np.asarray(t, dtype=float)
        signal = rng.normal(0.0, noise, size=t.shape) if noise else np.zeros_like(t)
        for pulse_time in pulse_times:
            signal += ((t >= pulse_time) & (t < pulse_time + pulse_width)).astype(float)
        if nchannels == 2:
            if signal.ndim == 0:
                return np.array([float(signal), float(signal)])
            return np.column_stack([signal, signal])
        return float(signal) if signal.ndim == 0 else signal

    return AudioClip(frame_function, duration=duration, fps=fps)


def color_clips(n, duration=2.0, size=SIZE, fps=VIDEO_FPS, colors=None):
    """Builds ``n`` solid color clips, identifiable by their center pixel."""
    colors = colors or COLORS
    return [
        ColorClip(size=size, color=colors[i % len(colors)], duration=duration).with_fps(
            fps
        )
        for i in range(n)
    ]


def center_color(clip, t):
    return clip.get_frame(t)[SIZE[1] // 2, SIZE[0] // 2].tolist()


def scalar(value):
    """Extracts a plain float from an audio frame (mono or stereo)."""
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


# ---------------------------------------------------------------------------
# Onset detection
# ---------------------------------------------------------------------------


def test_find_audio_onsets_detects_aligned_pulses():
    audio = pulse_audio(pulse_times=[0.5, 1.0, 1.5])
    onsets = find_audio_onsets(audio, window_duration=0.05, min_onset_distance=0.2)
    # 0 is always the first cut point; each pulse gives one nearby onset.
    assert onsets[0] == 0.0
    assert len(onsets) == 4
    detected = onsets[1:]
    for expected, found in zip([0.5, 1.0, 1.5], detected):
        assert found == pytest.approx(expected, abs=0.06)
    # Strictly increasing.
    assert all(b > a for a, b in zip(onsets, onsets[1:]))


def test_find_audio_onsets_respects_min_distance():
    # Two pulses closer than min_onset_distance must collapse into one onset.
    audio = pulse_audio(pulse_times=[0.5, 0.55])
    onsets = find_audio_onsets(audio, window_duration=0.02, min_onset_distance=0.3)
    assert onsets[0] == 0.0
    assert len(onsets) == 2
    assert onsets[1] == pytest.approx(0.5, abs=0.05)


def test_find_audio_onsets_robust_to_background_noise():
    clean = pulse_audio(pulse_times=[0.5, 1.0, 1.5])
    noisy = pulse_audio(pulse_times=[0.5, 1.0, 1.5], noise=0.01)
    kwargs = dict(window_duration=0.05, min_onset_distance=0.2)
    assert find_audio_onsets(noisy, **kwargs) == find_audio_onsets(clean, **kwargs)


def test_find_audio_onsets_silence_returns_only_zero():
    silent = pulse_audio(pulse_times=[])
    assert find_audio_onsets(silent, window_duration=0.05) == [0.0]


def test_find_audio_onsets_track_shorter_than_window():
    short = pulse_audio(pulse_times=[], duration=0.02)
    assert find_audio_onsets(short, window_duration=0.05) == [0.0]


def test_find_audio_onsets_stereo():
    stereo = pulse_audio(pulse_times=[0.5, 1.0], nchannels=2)
    onsets = find_audio_onsets(stereo, window_duration=0.05, min_onset_distance=0.2)
    assert onsets[0] == 0.0
    assert len(onsets) == 3


def test_find_audio_onsets_respects_max_duration():
    audio = pulse_audio(pulse_times=[0.5, 1.5], duration=2.0)
    onsets = find_audio_onsets(
        audio, window_duration=0.05, min_onset_distance=0.2, max_duration=1.0
    )
    assert all(t <= 1.0 + 1e-9 for t in onsets)
    # The 1.5s pulse must be ignored.
    assert max(onsets) < 1.2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(window_duration=0), "window_duration"),
        (dict(window_duration=-1), "window_duration"),
        (dict(energy_threshold=0), "energy_threshold"),
        (dict(min_onset_distance=-1), "min_onset_distance"),
        (dict(max_duration=0), "max_duration"),
    ],
)
def test_find_audio_onsets_validates_arguments(kwargs, message):
    audio = pulse_audio(pulse_times=[0.5])
    with pytest.raises(ValueError, match=message):
        find_audio_onsets(audio, **kwargs)


def test_find_audio_onsets_streams_without_loading_whole_track():
    # A one-hour silent track: the generator must be consumed only up to
    # max_duration, one window at a time.
    audio = pulse_audio(pulse_times=[], duration=3600.0, fps=44100)
    pulled_sizes = []
    original = audio.iter_chunks

    def counting_iter_chunks(*args, **kwargs):
        for chunk in original(*args, **kwargs):
            pulled_sizes.append(len(chunk))
            yield chunk

    audio.iter_chunks = counting_iter_chunks
    onsets = find_audio_onsets(audio, window_duration=0.05, max_duration=10.0)
    assert onsets == [0.0]
    # ~200 windows cover 10s; a full read would pull 72000 windows.
    assert len(pulled_sizes) < 210
    assert max(pulled_sizes) <= int(0.05 * 44100) + 1


# ---------------------------------------------------------------------------
# Montage assembly
# ---------------------------------------------------------------------------


def test_montage_cut_points_follow_pulses():
    audio = pulse_audio(pulse_times=[0.5, 1.0, 1.5])
    clips = color_clips(4)
    cut_times, montage = montage_on_onsets(
        audio, clips, duration=2.0, window_duration=0.05, min_shot_duration=0.3
    )
    assert cut_times[0] == 0.0
    assert cut_times[-1] == 2.0
    # Four shots => five boundaries, the inner ones sit near the pulses.
    assert len(cut_times) == 5
    for expected, found in zip([0.5, 1.0, 1.5], cut_times[1:-1]):
        assert found == pytest.approx(expected, abs=0.06)
    # Every shot respects the minimum length.
    gaps = np.diff(cut_times)
    assert all(gap >= 0.3 - 1e-9 for gap in gaps)
    assert montage.duration == pytest.approx(2.0)


def test_montage_frame_colors_match_clip_order():
    audio = pulse_audio(pulse_times=[0.5, 1.0, 1.5])
    clips = color_clips(4)
    cut_times, montage = montage_on_onsets(
        audio, clips, duration=2.0, window_duration=0.05, min_shot_duration=0.3
    )
    for index, (start, end) in enumerate(zip(cut_times[:-1], cut_times[1:])):
        midpoint = (start + end) / 2
        expected = list(COLORS[index])
        assert center_color(montage, midpoint) == expected


def test_montage_uniform_size_fps_and_exact_frame_count():
    audio = pulse_audio(pulse_times=[0.5, 1.0, 1.5])
    clips = color_clips(4)
    _, montage = montage_on_onsets(audio, clips, duration=2.0, min_shot_duration=0.3)
    assert tuple(montage.size) == SIZE
    assert montage.fps == VIDEO_FPS
    # 2.0s at 10 fps => exactly 20 frames.
    assert int(montage.duration * montage.fps) == 20


def test_montage_keeps_first_and_last_frames():
    audio = pulse_audio(pulse_times=[0.5, 1.0, 1.5])
    clips = color_clips(4)
    cut_times, montage = montage_on_onsets(
        audio, clips, duration=2.0, min_shot_duration=0.3
    )
    last_t = (int(montage.duration * montage.fps) - 1) / montage.fps
    # First frame is the first clip, last frame is the last used clip.
    assert center_color(montage, 0.0) == list(COLORS[0])
    assert center_color(montage, last_t) == list(COLORS[len(cut_times) - 2])


def test_montage_attaches_original_audio_for_exact_duration():
    audio = pulse_audio(pulse_times=[0.5, 1.0, 1.5], duration=2.0)
    clips = color_clips(4)
    _, montage = montage_on_onsets(audio, clips, duration=2.0, min_shot_duration=0.3)
    assert montage.audio is not None
    assert montage.audio.duration == pytest.approx(2.0)
    # The soundtrack carries the pulse (the source clips had no audio anyway).
    assert scalar(montage.audio.get_frame(0.51)) > 0


def test_montage_loops_short_clip_to_fill_shot():
    # A single pulse => two shots, both longer than the 0.4s source clips.
    audio = pulse_audio(pulse_times=[0.7])
    clips = color_clips(2, duration=0.4)
    cut_times, montage = montage_on_onsets(
        audio, clips, duration=2.0, min_shot_duration=0.3
    )
    assert montage.duration == pytest.approx(2.0)
    # The first clip is looped but keeps its own color for the whole shot.
    assert center_color(montage, (cut_times[0] + cut_times[1]) / 2) == list(COLORS[0])
    assert center_color(montage, 1.9) == list(COLORS[1])


def test_montage_trims_long_clip_to_shot_length():
    audio = pulse_audio(pulse_times=[0.5, 1.0, 1.5])
    clips = color_clips(4, duration=10.0)
    cut_times, montage = montage_on_onsets(
        audio, clips, duration=2.0, min_shot_duration=0.3
    )
    assert montage.duration == pytest.approx(2.0)
    assert all(clip.duration == 10.0 for clip in clips)
    assert cut_times[-1] == 2.0


def test_montage_loops_short_audio_to_target_duration():
    short_audio = pulse_audio(pulse_times=[0.5], duration=1.0)
    clips = color_clips(2)
    _, montage = montage_on_onsets(
        short_audio, clips, duration=2.0, min_shot_duration=0.3
    )
    assert montage.audio.duration == pytest.approx(2.0)
    # t = 1.5 in the looped track equals t = 0.5 of the 1s source.
    assert scalar(montage.audio.get_frame(1.5)) == pytest.approx(
        scalar(short_audio.get_frame(0.5))
    )


def test_montage_trims_long_audio_to_target_duration():
    long_audio = pulse_audio(pulse_times=[0.5, 1.0, 1.5], duration=5.0)
    clips = color_clips(4)
    _, montage = montage_on_onsets(
        long_audio, clips, duration=2.0, min_shot_duration=0.3
    )
    assert montage.audio.duration == pytest.approx(2.0)


def test_montage_silent_audio_produces_a_single_shot():
    silent = pulse_audio(pulse_times=[])
    clips = color_clips(1)
    cut_times, montage = montage_on_onsets(
        silent, clips, duration=2.0, min_shot_duration=0.3
    )
    assert cut_times == [0.0, 2.0]
    assert montage.duration == pytest.approx(2.0)
    assert center_color(montage, 1.0) == list(COLORS[0])


def test_montage_does_not_modify_source_clips():
    audio = pulse_audio(pulse_times=[0.5, 1.0, 1.5])
    clips = color_clips(4)
    snapshot = [(clip.duration, clip.size, clip.fps, clip.audio) for clip in clips]
    montage_on_onsets(audio, clips, duration=2.0, min_shot_duration=0.3)
    assert [
        (clip.duration, clip.size, clip.fps, clip.audio) for clip in clips
    ] == snapshot
    assert audio.duration == 2.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_montage_rejects_empty_clip_list():
    audio = pulse_audio(pulse_times=[0.5])
    with pytest.raises(ValueError, match="at least one video clip"):
        montage_on_onsets(audio, [], duration=2.0)


def test_montage_rejects_illegal_durations():
    audio = pulse_audio(pulse_times=[0.5])
    clips = color_clips(1)
    with pytest.raises(ValueError, match="strictly positive"):
        montage_on_onsets(audio, clips, duration=0)
    with pytest.raises(ValueError, match="strictly positive"):
        montage_on_onsets(audio, clips, duration=2.0, min_shot_duration=0)
    with pytest.raises(ValueError, match="min_shot_duration"):
        montage_on_onsets(audio, clips, duration=0.2, min_shot_duration=0.3)


def test_montage_rejects_audio_shorter_than_window():
    short_audio = pulse_audio(pulse_times=[], duration=0.02)
    clips = color_clips(1, duration=0.02)
    with pytest.raises(ValueError, match="shorter than one analysis window"):
        montage_on_onsets(
            short_audio,
            clips,
            duration=0.02,
            window_duration=0.05,
            min_shot_duration=0.01,
        )


def test_montage_rejects_not_enough_material():
    audio = pulse_audio(pulse_times=[0.5, 1.0, 1.5])
    with pytest.raises(ValueError, match="Not enough material"):
        montage_on_onsets(audio, color_clips(1), duration=2.0, min_shot_duration=0.3)


def test_montage_rejects_incompatible_sizes():
    audio = pulse_audio(pulse_times=[0.5])
    clips = color_clips(1) + [
        ColorClip(size=(64, 48), color=(1, 2, 3), duration=2.0).with_fps(VIDEO_FPS)
    ]
    with pytest.raises(ValueError, match="same size"):
        montage_on_onsets(audio, clips, duration=2.0)


def test_montage_rejects_incompatible_fps():
    audio = pulse_audio(pulse_times=[0.5])
    clips = color_clips(1) + [
        ColorClip(size=SIZE, color=(1, 2, 3), duration=2.0).with_fps(5)
    ]
    with pytest.raises(ValueError, match="same fps"):
        montage_on_onsets(audio, clips, duration=2.0)


def test_montage_rejects_clip_without_fps():
    audio = pulse_audio(pulse_times=[0.5])
    no_fps = ColorClip(size=SIZE, color=(1, 2, 3), duration=2.0)
    with pytest.raises(ValueError, match="fps"):
        montage_on_onsets(audio, [no_fps], duration=2.0)


def test_montage_rejects_audio_without_duration_or_fps():
    clips = color_clips(1)

    class NoFpsAudio:
        duration = 2.0
        fps = None

    class NoDurationAudio:
        duration = None
        fps = 1000

    with pytest.raises(ValueError, match="fps"):
        montage_on_onsets(NoFpsAudio(), clips, duration=2.0)
    with pytest.raises(ValueError, match="duration"):
        montage_on_onsets(NoDurationAudio(), clips, duration=2.0)


# ---------------------------------------------------------------------------
# Backward compatibility: existing APIs keep working unchanged.
# ---------------------------------------------------------------------------


def test_find_audio_period_still_available():
    # find_audio_period must keep its original signature and behavior.
    audio = pulse_audio(pulse_times=[0.5], duration=0.6)
    period = find_audio_period(audio, min_time=0.05, max_time=0.5, time_resolution=0.01)
    assert isinstance(period, float)
    assert period >= 0.0


def test_concatenate_videoclips_still_available():
    clips = color_clips(2, duration=0.5)
    result = concatenate_videoclips(clips, method="chain")
    assert result.duration == pytest.approx(1.0)
    assert result.fps == VIDEO_FPS
