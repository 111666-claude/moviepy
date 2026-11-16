"""Image sequencing clip tests meant to be run with pytest."""

import os
import shutil
import subprocess

import numpy as np
import pytest

from moviepy.audio.AudioClip import AudioClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip


FFPROBE_BINARY = shutil.which("ffprobe")

# One frame of tolerance when comparing read-back timestamps: FFMPEG's
# fallback frame rate for image sequences is 25 fps.
FRAME_TOLERANCE = 1.0 / 25


def test_1(util):
    images = []
    durations = []

    for i in range(5):
        durations.append(i)
        images.append("media/python_logo.png")
        durations.append(i)
        images.append("media/python_logo_upside_down.png")

    with ImageSequenceClip(images, durations=durations) as clip:
        assert clip.duration == sum(durations)
        clip.write_videofile(
            os.path.join(util.TMP_DIR, "ImageSequenceClip1.mp4"), fps=30, logger=None
        )


def test_2():
    images = []
    durations = []

    durations.append(1)
    images.append("media/python_logo.png")
    durations.append(2)
    images.append("media/matplotlib_demo1.png")

    # images are not the same size..
    with pytest.raises(Exception):
        ImageSequenceClip(images, durations=durations).close()


def _solid_color_frames(colors, size=(64, 48)):
    """Returns one solid-color numpy frame per color."""
    width, height = size
    return [np.full((height, width, 3), color, dtype="uint8") for color in colors]


def _pulse_track(duration):
    """A pulse train audio clip, with an extra pulse at the very end."""

    def make_frame(t):
        t = np.atleast_1d(t)
        return (0.5 * (((t % 0.4) < 0.05) | (t >= duration - 0.05))).astype(float)

    return AudioClip(make_frame, duration=duration, fps=44100)


def _ffprobe_video_timestamps(filename):
    """Returns (frame pts list, format duration) of a video file."""
    out = subprocess.check_output(
        [
            FFPROBE_BINARY,
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "packet=pts_time",
            "-of",
            "csv=p=0",
            filename,
        ]
    ).decode()
    pts = [float(line) for line in out.split()]
    duration = float(
        subprocess.check_output(
            [
                FFPROBE_BINARY,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                filename,
            ]
        ).decode()
    )
    return pts, duration


@pytest.mark.skipif(FFPROBE_BINARY is None, reason="ffprobe is not available")
@pytest.mark.parametrize("ext", ["mp4", "mkv"])
def test_vfr_export(util, ext):
    """Each image duration must become the actual display duration of a
    video frame in the exported VFR file.
    """
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    durations = [0.2, 1.0, 0.4]
    duration = sum(durations)
    filename = os.path.join(util.TMP_DIR, "ImageSequenceClip_vfr." + ext)

    clip = ImageSequenceClip(_solid_color_frames(colors), durations=durations)
    clip.audio = _pulse_track(duration)
    clip.write_videofile_vfr(filename, logger=None)

    # The video must hold exactly one frame per image: long still images
    # must not be duplicated into many identical frames.
    pts, file_duration = _ffprobe_video_timestamps(filename)
    assert len(pts) == len(durations)

    # Frame timestamps and total duration, within one frame of tolerance.
    expected_starts = [0.0, 0.2, 1.2]
    assert pts == pytest.approx(expected_starts, abs=FRAME_TOLERANCE)
    assert file_duration == pytest.approx(duration, abs=FRAME_TOLERANCE)

    # Colors are preserved, each image displayed during its own slot.
    with VideoFileClip(filename) as video:
        assert video.duration == pytest.approx(duration, abs=FRAME_TOLERANCE)
        for t, color in zip([0.1, 0.7, 1.4], colors):
            pixel = video.get_frame(t)[0, 0]
            assert list(pixel) == pytest.approx(list(color), abs=16)

    # The soundtrack stays synchronized: silence before the final pulse,
    # and the pulse is still audible at the very end of the file.
    with AudioFileClip(filename) as audio:
        assert audio.duration == pytest.approx(duration, abs=FRAME_TOLERANCE)
        sound = audio.to_soundarray(fps=8000)
        if sound.ndim > 1:
            sound = sound.mean(axis=1)
        silence = sound[int(1.3 * 8000) : int(1.5 * 8000)]
        tail = sound[int(1.55 * 8000) :]
        assert np.abs(silence).max() < 0.05
        assert np.abs(tail).max() > 0.1
    clip.close()


def test_vfr_export_without_audio(util):
    """VFR export also works with ``audio=False`` and from image files."""
    frames = _solid_color_frames([(10, 20, 30), (200, 100, 50)])
    clip = ImageSequenceClip(frames, durations=[0.5, 0.25])
    filename = os.path.join(util.TMP_DIR, "ImageSequenceClip_vfr_silent.mp4")
    clip.write_videofile_vfr(filename, audio=False, logger=None)
    with VideoFileClip(filename) as video:
        assert video.audio is None
        assert video.duration == pytest.approx(0.75, abs=FRAME_TOLERANCE)
        assert video.size == list(clip.size)
    clip.close()


@pytest.mark.parametrize(
    "durations",
    [
        [0.0, 0.5, 0.5],  # zero duration
        [0.5, -0.2, 0.9],  # negative duration
        [0.5, np.inf, 0.5],  # infinite duration
        [0.5, np.nan, 0.5],  # NaN duration
    ],
)
def test_vfr_export_invalid_durations(util, durations):
    frames = _solid_color_frames([(0, 0, 0)] * 3, size=(16, 16))
    clip = ImageSequenceClip(frames, durations=durations)
    with pytest.raises(ValueError, match="duration"):
        clip.write_videofile_vfr(
            os.path.join(util.TMP_DIR, "ImageSequenceClip_bad.mp4"), logger=None
        )
    clip.close()


def test_vfr_export_unsupported_container(util):
    frames = _solid_color_frames([(0, 0, 0)] * 2, size=(16, 16))
    clip = ImageSequenceClip(frames, durations=[0.5, 0.5])
    with pytest.raises(ValueError, match="container"):
        clip.write_videofile_vfr(
            os.path.join(util.TMP_DIR, "ImageSequenceClip_bad.avi"), logger=None
        )
    clip.close()


def test_vfr_export_ffmpeg_failure(util):
    frames = _solid_color_frames([(0, 0, 0)] * 2, size=(16, 16))
    clip = ImageSequenceClip(frames, durations=[0.5, 0.5])
    with pytest.raises(IOError, match="FFMPEG"):
        clip.write_videofile_vfr(
            os.path.join(util.TMP_DIR, "ImageSequenceClip_bad_codec.mp4"),
            codec="this_codec_does_not_exist",
            logger=None,
        )
    clip.close()


if __name__ == "__main__":
    pytest.main()
