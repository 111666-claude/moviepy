"""Tests for the trajectory-based ``Stabilize`` video effect."""

import numpy as np

import pytest

from moviepy import AudioClip, VideoClip, vfx


WIDTH, HEIGHT, FPS, DURATION = 40, 30, 10, 1.0

# Known integer camera translations per frame. A single bright world point at
# (10, 8) recorded with a camera shift of (dx, dy) lights up pixel
# (10 + dx, 8 + dy).
FRAME_SHIFTS = [
    (2, 1),
    (-3, 2),
    (1, -1),
    (0, 0),
    (3, -2),
    (-1, 1),
    (2, 2),
    (-2, -1),
    (1, 0),
    (0, 1),
]


def make_translated_clip(
    width=WIDTH,
    height=HEIGHT,
    fps=FPS,
    duration=DURATION,
    shifts=FRAME_SHIFTS,
    marker=(10, 8),
):
    """Build a synthetic clip whose content moves by known integer shifts."""

    def make_frame(t):
        index = min(int(round(t * fps)), len(shifts) - 1)
        dx, dy = shifts[index]
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[marker[1] + dy, marker[0] + dx] = (255, 255, 255)
        return frame

    return VideoClip(make_frame, duration=duration).with_fps(fps)


def trajectory_from_shifts(shifts=FRAME_SHIFTS, fps=FPS, duration=DURATION):
    """Keyframes (time, dx, dy) covering the whole clip duration."""
    points = [
        (index / fps, float(dx), float(dy)) for index, (dx, dy) in enumerate(shifts)
    ]
    points.append((duration, float(shifts[-1][0]), float(shifts[-1][1])))
    return points


def frame_marker_center(frame):
    """Mean position of the bright marker pixels in one frame."""
    ys, xs = np.nonzero(frame[:, :, 0])
    return float(xs.mean()), float(ys.mean())


def test_stabilize_locks_target_center_with_known_translation():
    """With smoothing disabled every frame returns to the anchor position."""
    clip = make_translated_clip()
    trajectory = trajectory_from_shifts()

    effect = vfx.Stabilize(trajectory=trajectory, smoothing=0)
    stabilized = effect.apply(clip.copy())

    # Residuals are measured against the first frame (2, 1): max horizontal
    # residual is 5, max vertical residual is 3.
    assert effect.crop_margins == (5, 5, 3, 3)
    assert effect.output_size == (WIDTH - 10, HEIGHT - 6)
    assert stabilized.size == (WIDTH - 10, HEIGHT - 6)

    left, _, top, _ = effect.crop_margins
    # The lock target is the anchor frame's recorded marker position
    # (10 + dx0, 8 + dy0) = (12, 9), minus the crop margins.
    expected_center = (10 + FRAME_SHIFTS[0][0] - left, 8 + FRAME_SHIFTS[0][1] - top)
    assert expected_center == (7, 6)

    # Exact integer shifts must put the single marker on the exact same pixel
    # in every output frame.
    for index in range(len(FRAME_SHIFTS)):
        frame = stabilized.get_frame(index / FPS)
        ys, xs = np.nonzero(frame[:, :, 0])
        assert len(xs) == 1
        assert (int(xs[0]), int(ys[0])) == expected_center
        # Exactly one bright RGB pixel, no replicated border.
        assert (frame == 255).sum() == 3


def test_stabilize_fixed_size_and_crop_parameters_on_clip():
    """The clip exposes crop parameters and keeps a constant frame size."""
    clip = make_translated_clip()
    stabilized = clip.with_effects(
        [vfx.Stabilize(trajectory=trajectory_from_shifts(), smoothing=0)]
    )

    assert stabilized.stabilize_crop_margins == (5, 5, 3, 3)
    assert stabilized.stabilize_output_size == (WIDTH - 10, HEIGHT - 6)
    assert stabilized.stabilize_smoothed_trajectory.shape == (10, 3)
    assert stabilized.size == (WIDTH - 10, HEIGHT - 6)

    rng = np.random.default_rng(42)
    for _ in range(25):
        t = float(rng.uniform(0.0, DURATION - 1e-9))
        frame = stabilized.get_frame(t)
        assert frame.shape == (HEIGHT - 6, WIDTH - 10, 3)
        # Sub-pixel shifts interpolate the marker but never produce an empty
        # frame or an out-of-frame sample.
        assert frame.max() > 0


def test_stabilize_keeps_duration_fps_and_audio():
    """Audio, duration and fps are not modified by the effect."""
    clip = make_translated_clip()
    audio = AudioClip(
        lambda t: np.sin(440 * 2 * np.pi * t), duration=DURATION, fps=44100
    )
    clip = clip.with_audio(audio)

    stabilized = clip.with_effects(
        [vfx.Stabilize(trajectory=trajectory_from_shifts(), smoothing=0)]
    )

    assert stabilized.fps == FPS
    assert stabilized.duration == pytest.approx(DURATION)
    assert stabilized.audio is not None
    assert stabilized.audio.duration == pytest.approx(DURATION)
    for t in (0.0, 0.37, 0.9):
        assert np.allclose(stabilized.audio.get_frame(t), audio.get_frame(t))


def test_stabilize_applies_same_transform_to_mask():
    """The mask is warped with the same per-frame spatial transform."""
    clip = make_translated_clip().with_mask()
    stabilized = clip.with_effects(
        [vfx.Stabilize(trajectory=trajectory_from_shifts(), smoothing=0)]
    )

    assert stabilized.mask is not None
    assert stabilized.mask.size == stabilized.size
    for t in (0.1, 0.45, 0.9):
        mask_frame = stabilized.mask.get_frame(t)
        assert mask_frame.shape == (stabilized.h, stabilized.w)
        assert np.allclose(mask_frame, 1.0)


def test_stabilize_interpolates_missing_keyframes():
    """A sparse trajectory gives the same frames as its dense interpolation."""
    clip = make_translated_clip()
    sparse = [
        (0.0, 0.0, 0.0),
        (0.5, 4.0, -2.0),
        (1.0, -2.0, 2.0),
    ]
    dense = [
        (
            index / 10,
            float(np.interp(index / 10, [0, 0.5, 1], [0, 4, -2])),
            float(np.interp(index / 10, [0, 0.5, 1], [0, -2, 2])),
        )
        for index in range(11)
    ]

    sparse_clip = clip.with_effects([vfx.Stabilize(trajectory=sparse)])
    dense_clip = clip.with_effects([vfx.Stabilize(trajectory=dense)])

    assert sparse_clip.stabilize_crop_margins == dense_clip.stabilize_crop_margins
    for t in (0.07, 0.27, 0.63, 0.91):
        assert np.allclose(sparse_clip.get_frame(t), dense_clip.get_frame(t))


def test_stabilize_smoothing_reduces_jitter():
    """A moving-average trajectory leaves less residual motion than the raw
    jitter, while the output size stays constant."""
    clip = make_translated_clip()
    stabilized = clip.with_effects(
        [vfx.Stabilize(trajectory=trajectory_from_shifts(), smoothing=0.3)]
    )

    raw_centers = [(10 + dx, 8 + dy) for dx, dy in FRAME_SHIFTS]
    smoothed_centers = [
        frame_marker_center(stabilized.get_frame(index / FPS))
        for index in range(len(FRAME_SHIFTS))
    ]

    assert np.ptp(smoothed_centers, axis=0).max() < np.ptp(raw_centers, axis=0).max()
    for index in range(len(FRAME_SHIFTS)):
        assert stabilized.get_frame(index / FPS).shape == (
            stabilized.h,
            stabilized.w,
            3,
        )


def test_stabilize_explicit_border_crop():
    """User-provided margins are honored and returned as the actual crop."""
    clip = make_translated_clip()
    effect = vfx.Stabilize(
        trajectory=trajectory_from_shifts(),
        smoothing=0,
        border_crop=(6, 6, 4, 4),
    )
    stabilized = effect.apply(clip.copy())

    assert effect.crop_margins == (6, 6, 4, 4)
    assert stabilized.size == (WIDTH - 12, HEIGHT - 8)
    assert stabilized.stabilize_crop_margins == (6, 6, 4, 4)


def test_stabilize_random_times_stay_inside_source():
    """Every frame at an arbitrary time is fully covered by the source."""
    clip = make_translated_clip()
    stabilized = clip.with_effects(
        [vfx.Stabilize(trajectory=trajectory_from_shifts(), smoothing=0)]
    )
    rng = np.random.default_rng(7)
    for _ in range(40):
        t = float(rng.uniform(0.0, DURATION))
        frame = stabilized.get_frame(t)
        assert frame.shape == (stabilized.h, stabilized.w, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trajectory": []},
        {"trajectory": None},
        {"trajectory": [(0, 0, 0), (0.5, np.nan, 1), (1, 0, 0)]},
        {"trajectory": [(0, 0, 0), (0.5, 2, np.inf), (1, 0, 0)]},
        {"trajectory": [(1.0, 0, 0), (0.5, 2, 1), (0, 0, 0)]},
        {"trajectory": [(0, 0, 0), (0.5, 2, 1), (0.5, 1, 1), (1, 0, 0)]},
        {"trajectory": [(0.2, 0, 0), (0.5, 2, 1), (1, 0, 0)]},
        {"trajectory": [(0, 0, 0), (0.5, 2, 1), (0.9, 0, 0)]},
        {"trajectory": [(0, 0), (0.5, 2), (1, 0)]},
        {"trajectory": [(0, 0, 0, 1), (0.5, 2, 1, 1), (1, 0, 0, 1)]},
        {"trajectory": [(0, "x", 0), (0.5, 2, 1), (1, 0, 0)]},
        {"trajectory": [(0.0, 1.0, 1.0)]},
    ],
)
def test_stabilize_rejects_invalid_trajectory(kwargs):
    """Non-finite coordinates, reversed times, empty and malformed trajectories
    all raise a clear ``ValueError``."""
    clip = make_translated_clip()
    with pytest.raises(ValueError):
        clip.with_effects([vfx.Stabilize(**kwargs)])


def test_stabilize_rejects_unusable_border_crop():
    """Margins that cannot contain the compensation, or leave no frame area,
    are rejected instead of producing black borders."""
    clip = make_translated_clip()
    trajectory = trajectory_from_shifts()

    with pytest.raises(ValueError):
        clip.with_effects(
            [vfx.Stabilize(trajectory=trajectory, border_crop=(0, 0, 5, 5))]
        )
    with pytest.raises(ValueError):
        clip.with_effects(
            [vfx.Stabilize(trajectory=trajectory, border_crop=(20, 20, 2, 2))]
        )
    with pytest.raises(ValueError):
        clip.with_effects(
            [vfx.Stabilize(trajectory=trajectory, border_crop=(-1, 2, 2, 2))]
        )
    with pytest.raises(ValueError):
        clip.with_effects([vfx.Stabilize(trajectory=trajectory, border_crop=(1, 2, 3))])
    with pytest.raises(ValueError):
        clip.with_effects([vfx.Stabilize(trajectory=trajectory, smoothing=-1)])
    with pytest.raises(ValueError):
        clip.with_effects([vfx.Stabilize(trajectory=trajectory, interpolation="weird")])


def test_stabilize_requires_fps():
    """A clip without fps needs an explicit ``fps`` argument."""
    clip = VideoClip(lambda t: np.zeros((20, 30, 3), np.uint8), duration=1.0)
    clip.fps = None
    trajectory = [(0.0, 0.0, 0.0), (0.5, 2.0, 1.0), (1.0, -1.0, 0.0)]

    with pytest.raises(ValueError):
        clip.with_effects([vfx.Stabilize(trajectory=trajectory)])

    stabilized = clip.with_effects([vfx.Stabilize(trajectory=trajectory, fps=10)])
    assert stabilized.size == (26, 18)
