"""Deterministic tests for the ``AudioDuck`` audio effect.

All the signals are synthesized from constant amplitudes and square
voice-over gates, so the expected gains and timing can be derived
analytically instead of being read from a media file.
"""

import numpy as np

import pytest

from moviepy import AudioClip, CompositeAudioClip, afx
from moviepy.video.VideoClip import VideoClip


FPS = 1000
DURATION = 3.0
TARGET_GAIN = 0.2
THRESHOLD = 0.05
ATTACK = 0.1
RELEASE = 0.2
WINDOW = 0.02


def gate(t, segments):
    """Return ``True`` where ``t`` falls into a voice-over segment."""
    t = np.asarray(t, dtype=float)
    active = np.zeros(t.shape, dtype=bool)
    for start, end in segments:
        active |= (t >= start) & (t < end)
    return active


def make_voice_clip(segments, duration=DURATION, fps=FPS, amplitude=0.5):
    """Build a mono voice-over active on ``segments``, silent elsewhere."""

    def frame_function(t):
        if np.isscalar(t):
            return float(amplitude) if bool(gate(t, segments)) else 0.0
        return np.where(gate(t, segments), amplitude, 0.0)

    return AudioClip(frame_function, duration=duration, fps=fps)


def make_music_clip(duration=DURATION, fps=FPS, stereo=False, value=1.0):
    """Build a constant-amplitude clip whose output equals the envelope."""
    if stereo:

        def frame_function(t):
            if np.isscalar(t):
                return np.array([value, value])
            return np.full((len(np.asarray(t)), 2), value)

    else:

        def frame_function(t):
            if np.isscalar(t):
                return float(value)
            return np.full(np.asarray(t).shape, value, dtype=float)

    return AudioClip(frame_function, duration=duration, fps=fps)


def duck(music, voice, **kwargs):
    """Apply AudioDuck and render the whole clip at the music fps."""
    options = dict(
        target_gain=TARGET_GAIN,
        threshold=THRESHOLD,
        attack=ATTACK,
        release=RELEASE,
        analysis_window=WINDOW,
    )
    options.update(kwargs)
    ducked = music.with_effects([afx.AudioDuck(voice, **options)])
    times = np.arange(0.0, music.duration, 1.0 / music.fps)
    return ducked, ducked.to_soundarray(times, fps=music.fps)


def test_ducking_reaches_target_gain():
    """The music settles on ``target_gain`` inside a long speech segment."""
    segments = [(0.4, 1.6)]
    music = make_music_clip()
    voice = make_voice_clip(segments)

    # A 1-sample window removes the window delay from the timing budget, and
    # a short attack lets the gain fully settle before the assertions.
    _, output = duck(music, voice, attack=0.02, analysis_window=0.001)

    # 0.5 s after the gate opens the transient is long finished.
    assert output[int(0.9 * FPS)] == pytest.approx(TARGET_GAIN, abs=1e-3)
    assert output[int(1.5 * FPS)] == pytest.approx(TARGET_GAIN, abs=1e-3)
    # The gain never goes below the configured target.
    assert output.min() >= TARGET_GAIN - 1e-12


def test_attack_time_constant():
    """The first ducked sample follows the one-pole attack law."""
    segments = [(0.5, 1.5)]
    music = make_music_clip()
    voice = make_voice_clip(segments)

    ducked, output = duck(music, voice, attack=0.1, release=0.1, analysis_window=0.001)

    # The window is 1 sample wide, so the gate and the envelope switch
    # together at t = 0.5 s.
    first_ducked = int(0.5 * FPS)
    assert output[first_ducked - 1] == pytest.approx(1.0)

    attack_factor = 1.0 - np.exp(-1.0 / (0.1 * FPS))
    expected_first = TARGET_GAIN + (1.0 - TARGET_GAIN) * (1.0 - attack_factor)
    assert output[first_ducked] == pytest.approx(expected_first, abs=1e-12)

    # After one time constant most of the distance towards target is covered.
    assert output[first_ducked + int(0.1 * FPS)] < 0.55
    # And the scalar get_frame path agrees with the array-rendered path.
    assert ducked.get_frame(0.5) == pytest.approx(expected_first, abs=1e-12)


def test_release_time_constant_and_tail_recovery():
    """The gain climbs back to 1 after speech, including at the clip end."""
    segments = [(0.2, 0.8)]
    music = make_music_clip()
    voice = make_voice_clip(segments)

    _, output = duck(music, voice, attack=0.05, release=0.2, analysis_window=0.001)

    # Speech ends at 0.8 s; the 1-sample window clears the energy immediately.
    release_start = int(0.8 * FPS)
    release_factor = 1.0 - np.exp(-1.0 / (0.2 * FPS))
    expected_first = 1.0 + (TARGET_GAIN - 1.0) * (1.0 - release_factor)
    assert output[release_start] == pytest.approx(expected_first, rel=1e-3)

    # Monotonic recovery during the pause.
    pause = output[release_start : release_start + int(0.5 * FPS)]
    assert np.all(np.diff(pause) > 0)

    # The release is still applied on samples at the very end of the clip.
    assert output[-1] > output[release_start]
    assert output[-1] < 1.0


def test_two_narrations_recover_in_the_pause():
    """Two segments produce two ducking bands with recovery in between."""
    segments = [(0.4, 1.0), (1.8, 2.4)]
    music = make_music_clip()
    voice = make_voice_clip(segments)

    _, output = duck(music, voice, attack=0.05, release=0.1)

    # Mid-pause the music is back close to its original level.
    assert output[int(1.5 * FPS)] == pytest.approx(1.0, abs=2e-2)
    # Then ducks again for the second narration.
    assert output[int(2.1 * FPS)] == pytest.approx(TARGET_GAIN, abs=1e-2)


def test_silent_voiceover_leaves_music_unchanged():
    """A silent or sub-threshold narration must not alter the music."""
    music = make_music_clip()
    silent = make_voice_clip([])

    ducked, output = duck(music, silent)
    np.testing.assert_allclose(output, 1.0)

    # The frame function is not wrapped when no ducking is required.
    assert ducked.frame_function == music.frame_function

    # A quiet but non-zero narration below the threshold is also ignored.
    quiet = make_voice_clip([(0.0, 2.0)], amplitude=THRESHOLD / 2)
    _, quiet_output = duck(music, quiet)
    np.testing.assert_allclose(quiet_output, 1.0)


def test_threshold_distinguishes_speech_from_silence():
    """Windows above the threshold duck, windows below it do not."""
    # 0.03 RMS stays below the default 0.05 threshold.
    quiet = make_voice_clip([(0.0, 2.0)], amplitude=0.03)
    loud = make_voice_clip([(0.0, 2.0)], amplitude=0.3)
    music = make_music_clip()

    _, quiet_output = duck(music, quiet)
    assert np.allclose(quiet_output, 1.0)

    _, loud_output = duck(music, loud)
    assert loud_output[int(1.0 * FPS)] == pytest.approx(TARGET_GAIN, abs=1e-3)


def test_stereo_music_and_stereo_voiceover():
    """One envelope attenuates every music channel identically."""
    segments = [(0.4, 1.6)]

    def stereo_voice(t):
        if np.isscalar(t):
            value = 0.5 if bool(gate(t, segments)) else 0.0
            return np.array([value, 0.8 * value])
        active = gate(t, segments)
        left = np.where(active, 0.5, 0.0)
        right = np.where(active, 0.4, 0.0)
        return np.array([left, right]).T.copy(order="C")

    music = make_music_clip(stereo=True)
    voice = AudioClip(stereo_voice, duration=DURATION, fps=FPS)

    _, output = duck(music, voice, attack=0.02, analysis_window=0.001)

    assert output.shape == (int(DURATION * FPS), 2)
    # Both channels use the very same gain.
    np.testing.assert_array_equal(output[:, 0], output[:, 1])
    assert output[int(1.0 * FPS), 0] == pytest.approx(TARGET_GAIN, abs=1e-3)
    # No ducking before the narration starts.
    np.testing.assert_allclose(output[int(0.1 * FPS)], [1.0, 1.0])


def test_voiceover_start_offset_and_clipped_end():
    """Only the region overlapping the music is ducked."""
    music = make_music_clip(duration=2.0)
    # Active from 0.0 to 1.2 s of narration time, placed at 0.5 s on the
    # timeline, so it runs until 1.7 s and is clipped by the music at 2.0 s.
    voice = make_voice_clip([(0.0, 1.2)], duration=1.2).with_start(0.5)

    _, output = duck(music, voice, attack=0.02, release=0.1)

    assert output[int(0.3 * FPS)] == pytest.approx(1.0)
    assert output[int(0.8 * FPS)] == pytest.approx(TARGET_GAIN, abs=1e-3)
    # Speech is still active at the end of the music, the gain stays low.
    assert output[int(1.6 * FPS)] < 0.3
    assert len(output) == int(2.0 * FPS)


def test_voiceover_started_before_the_music():
    """Speech already active at t=0 must leave the music pre-ducked."""
    music = make_music_clip(duration=2.0)
    # Narration starts 0.3 s before the music and is active for its first
    # second, i.e. on the timeline from -0.3 s to 0.7 s.
    voice = make_voice_clip([(0.0, 1.0)], duration=1.5).with_start(-0.3)

    _, output = duck(music, voice, attack=0.02, release=0.1, analysis_window=0.005)

    # No attack replay at the start: the first sample is already ducked.
    assert output[0] == pytest.approx(TARGET_GAIN)
    assert output[int(0.5 * FPS)] == pytest.approx(TARGET_GAIN, abs=1e-3)
    # And it recovers once that narration segment ends.
    assert output[int(1.2 * FPS)] == pytest.approx(1.0, abs=1e-2)


def test_different_voiceover_sampling_rate():
    """A narration with another fps is resampled on the music timeline."""
    music = make_music_clip()
    voice = make_voice_clip([(0.4, 1.6)], fps=300)

    _, output = duck(music, voice)

    assert output[int(1.0 * FPS)] == pytest.approx(TARGET_GAIN, abs=1e-2)
    assert output[int(0.1 * FPS)] == pytest.approx(1.0)


def test_very_short_voiceover_segment():
    """A few-millisecond narration still triggers a duck and a release."""
    music = make_music_clip()
    voice = make_voice_clip([(0.5, 0.505)])

    _, output = duck(music, voice, attack=0.0, release=0.05, analysis_window=0.002)

    assert output[int(0.502 * FPS)] == pytest.approx(TARGET_GAIN, abs=1e-3)
    # The music recovers afterwards.
    assert output[int(0.7 * FPS)] == pytest.approx(1.0, abs=5e-2)


def test_instantaneous_attack_and_release():
    """Zero time constants give sample-accurate switching."""
    segments = [(0.5, 1.5)]
    music = make_music_clip()
    voice = make_voice_clip(segments)

    _, output = duck(music, voice, attack=0.0, release=0.0, analysis_window=0.001)

    assert output[int(0.5 * FPS) - 1] == pytest.approx(1.0)
    assert output[int(0.5 * FPS)] == pytest.approx(TARGET_GAIN)
    assert output[int(1.5 * FPS)] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": 0}, "threshold"),
        ({"threshold": -0.1}, "threshold"),
        ({"target_gain": -0.1}, "target_gain"),
        ({"target_gain": 1.5}, "target_gain"),
        ({"attack": -0.2}, "attack"),
        ({"release": -0.2}, "release"),
        ({"analysis_window": -0.01}, "analysis_window"),
    ],
)
def test_invalid_parameters_raise(kwargs, message):
    voice = make_voice_clip([(0.0, 1.0)])
    with pytest.raises(ValueError, match=message):
        afx.AudioDuck(voice, **kwargs)


def test_missing_and_negative_durations_raise():
    music = make_music_clip()
    voice = make_voice_clip([(0.0, 1.0)])

    music_without_duration = AudioClip(lambda t: 1.0, fps=FPS)
    with pytest.raises(ValueError, match="music clip"):
        music_without_duration.with_effects([afx.AudioDuck(voice)])

    voice_without_duration = AudioClip(lambda t: 0.0, fps=FPS)
    with pytest.raises(ValueError, match="voice-over clip"):
        music.with_effects([afx.AudioDuck(voice_without_duration)])

    zero_music = AudioClip(lambda t: 1.0, duration=0, fps=FPS)
    with pytest.raises(ValueError, match="music clip"):
        zero_music.with_effects([afx.AudioDuck(voice)])

    zero_voice = AudioClip(lambda t: 0.0, duration=0, fps=FPS)
    with pytest.raises(ValueError, match="voice-over clip"):
        music.with_effects([afx.AudioDuck(zero_voice)])


def test_music_without_fps_raises():
    voice = make_voice_clip([(0.0, 1.0)])
    music = AudioClip(lambda t: 1.0, duration=DURATION)
    with pytest.raises(ValueError, match="fps"):
        music.with_effects([afx.AudioDuck(voice)])


def test_unity_target_gain_is_a_noop():
    """``target_gain=1`` means the music is never modified."""
    music = make_music_clip()
    voice = make_voice_clip([(0.0, 2.0)])

    ducked, output = duck(music, voice, target_gain=1.0)
    np.testing.assert_allclose(output, 1.0)
    assert ducked.frame_function == music.frame_function


def test_works_with_composite_audio_clip():
    """The ducked music composes with the narration into their sum."""
    segments = [(0.0, 0.6)]

    def stereo_voice(t):
        if np.isscalar(t):
            value = 0.5 if bool(gate(t, segments)) else 0.0
            return np.array([value, value])
        active = gate(t, segments)
        value = np.where(active, 0.5, 0.0)
        return np.array([value, value]).T.copy(order="C")

    def stereo_music(t):
        if np.isscalar(t):
            return np.array([1.0, 1.0])
        return np.ones((len(np.asarray(t)), 2))

    music = AudioClip(stereo_music, duration=2.0, fps=FPS)
    voice = AudioClip(stereo_voice, duration=0.6, fps=FPS).with_start(0.5)
    ducked = music.with_effects(
        [
            afx.AudioDuck(
                voice,
                target_gain=TARGET_GAIN,
                threshold=THRESHOLD,
                attack=0.0,
                release=0.05,
                analysis_window=0.005,
            )
        ]
    )

    composite = CompositeAudioClip([ducked, voice])
    rendered = composite.to_soundarray(fps=FPS)

    assert rendered.shape == (int(2.0 * FPS), 2)
    # During speech: ducked music (0.2) + narration (0.5).
    np.testing.assert_allclose(rendered[int(0.7 * FPS)], [0.7, 0.7], atol=1e-3)
    # Before the narration: untouched music only.
    np.testing.assert_allclose(rendered[int(0.2 * FPS)], [1.0, 1.0])


def test_works_on_video_clip_audio():
    """Ducking applies to a video clip's audio but not its frames."""
    segments = [(0.0, 0.5)]
    voice = make_voice_clip(segments, duration=0.5)
    audio = make_music_clip(duration=1.0)

    video = VideoClip(
        lambda t: np.zeros((4, 4, 3), dtype=np.uint8), duration=1.0
    ).with_audio(audio)

    ducked_video = video.with_effects(
        [
            afx.AudioDuck(
                voice,
                threshold=THRESHOLD,
                attack=0.01,
                release=0.05,
                analysis_window=0.001,
            )
        ]
    )

    assert ducked_video.get_frame(0.0).shape == (4, 4, 3)
    rendered_audio = ducked_video.audio.to_soundarray(fps=FPS)
    assert rendered_audio[int(0.25 * FPS)] == pytest.approx(TARGET_GAIN, abs=1e-2)

    # A video clip without audio is simply returned unchanged.
    silent_video = VideoClip(
        lambda t: np.zeros((4, 4, 3), dtype=np.uint8), duration=1.0
    )
    result = silent_video.with_effects([afx.AudioDuck(voice)])
    assert isinstance(result, VideoClip)


def test_time_string_parameters_accepted():
    """Time constants accept the library's string and tuple expressions."""
    effect = afx.AudioDuck(
        make_voice_clip([(0.0, 1.0)]),
        attack="00:00:00.2",
        release=(0, 0.5),
        analysis_window="00:00:00,05",
    )
    assert effect.attack == pytest.approx(0.2)
    assert effect.release == pytest.approx(0.5)
    assert effect.analysis_window == pytest.approx(0.05)
