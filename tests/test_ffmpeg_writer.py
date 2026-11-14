"""FFmpeg writer tests of moviepy."""

import multiprocessing
import os

import numpy as np
from PIL import Image

import pytest

from moviepy import *
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
from moviepy.video.io.ffmpeg_writer import ffmpeg_write_image, ffmpeg_write_video
from moviepy.video.io.gif_writers import write_gif_with_imageio
from moviepy.video.tools.drawing import color_gradient


@pytest.mark.parametrize(
    "with_mask",
    (False, True),
    ids=("with_mask=False", "with_mask=True"),
)
@pytest.mark.parametrize(
    "write_logfile",
    (False, True),
    ids=("write_logfile=False", "write_logfile=True"),
)
@pytest.mark.parametrize(
    ("codec", "is_valid_codec", "ext"),
    (
        pytest.param(
            "libcrazyfoobar", False, ".mp4", id="codec=libcrazyfoobar-ext=.mp4"
        ),
        pytest.param(None, True, ".mp4", id="codec=default-ext=.mp4"),
        pytest.param("libtheora", False, ".avi", id="codec=libtheora-ext=.mp4"),
    ),
)
@pytest.mark.parametrize(
    "bitrate",
    (None, "5000k"),
    ids=("bitrate=None", "bitrate=5000k"),
)
@pytest.mark.parametrize(
    "threads",
    (None, multiprocessing.cpu_count()),
    ids=("threads=None", "threads=multiprocessing.cpu_count()"),
)
def test_ffmpeg_write_video(
    util,
    codec,
    is_valid_codec,
    ext,
    write_logfile,
    with_mask,
    bitrate,
    threads,
):
    filename = os.path.join(util.TMP_DIR, f"moviepy_ffmpeg_write_video{ext}")
    if os.path.isfile(filename):
        try:
            os.remove(filename)
        except PermissionError:
            pass

    logfile_name = filename + ".log"
    if os.path.isfile(logfile_name):
        os.remove(logfile_name)

    clip = BitmapClip([["R"], ["G"], ["B"]], fps=10).with_duration(0.3)
    if with_mask:
        clip = clip.with_mask(
            BitmapClip([["W"], ["O"], ["O"]], fps=10, is_mask=True).with_duration(0.3)
        )

    kwargs = dict(
        logger=None,
        write_logfile=write_logfile,
    )
    if codec is not None:
        kwargs["codec"] = codec
    if bitrate is not None:
        kwargs["bitrate"] = bitrate
    if threads is not None:
        kwargs["threads"] = threads

    ffmpeg_write_video(clip, filename, 10, **kwargs)

    if is_valid_codec:
        assert os.path.isfile(filename)

        final_clip = VideoFileClip(filename)

        r, g, b = final_clip.get_frame(0)[0][0]
        assert r == 254
        assert g == 0
        assert b == 0

        r, g, b = final_clip.get_frame(0.1)[0][0]
        assert r == (0 if not with_mask else 1)
        assert g == (255 if not with_mask else 1)
        assert b == 1

        r, g, b = final_clip.get_frame(0.2)[0][0]
        assert r == 0
        assert g == 0
        assert b == (255 if not with_mask else 0)

    if write_logfile:
        assert os.path.isfile(logfile_name)


@pytest.mark.parametrize(
    ("size", "logfile", "pixel_format", "expected_result"),
    (
        pytest.param(
            (5, 1),
            False,
            None,
            [[(0, 255, 0), (51, 204, 0), (102, 153, 0), (153, 101, 0), (204, 50, 0)]],
            id="size=(5, 1)",
        ),
        pytest.param(
            (2, 1), False, None, [[(0, 255, 0), (51, 204, 0)]], id="size=(2, 1)"
        ),
        pytest.param(
            (2, 1), True, None, [[(0, 255, 0), (51, 204, 0)]], id="logfile=True"
        ),
        pytest.param(
            (2, 1),
            False,
            "invalid",
            (OSError, "MoviePy error: FFMPEG encountered the following error"),
            id="pixel_format=invalid-OSError",
        ),
    ),
)
def test_ffmpeg_write_image(util, size, logfile, pixel_format, expected_result):
    filename = os.path.join(util.TMP_DIR, "moviepy_ffmpeg_write_image.png")
    if os.path.isfile(filename):
        try:
            os.remove(filename)
        except PermissionError:
            pass

    image_array = color_gradient(
        size,
        (0, 0),
        p2=(5, 0),
        color_1=(255, 0, 0),
        color_2=(0, 255, 0),
    )

    if hasattr(expected_result[0], "__traceback__"):
        with pytest.raises(expected_result[0]) as exc:
            ffmpeg_write_image(
                filename,
                image_array,
                logfile=logfile,
                pixel_format=pixel_format,
            )
        assert expected_result[1] in str(exc.value)
        return
    else:
        ffmpeg_write_image(
            filename,
            image_array,
            logfile=logfile,
            pixel_format=pixel_format,
        )

    assert os.path.isfile(filename)

    if logfile:
        assert os.path.isfile(filename + ".log")
        os.remove(filename + ".log")

    im = Image.open(filename, mode="r")
    for i in range(im.width):
        for j in range(im.height):
            assert im.getpixel((i, j)) == expected_result[j][i]


@pytest.mark.parametrize("loop", (0, 2), ids=("loop=0", "loop=2"))
@pytest.mark.parametrize("clip_class", ("BitmapClip", "ColorClip"))
@pytest.mark.parametrize(
    "with_mask", (False, True), ids=("with_mask=False", "with_mask=True")
)
def test_write_gif(util, clip_class, loop, with_mask):
    filename = os.path.join(util.TMP_DIR, "moviepy_write_gif.gif")
    if os.path.isfile(filename):
        try:
            os.remove(filename)
        except PermissionError:
            pass

    fps = 10

    if clip_class == "BitmapClip":
        original_clip = BitmapClip([["R"], ["G"], ["B"]], fps=fps).with_duration(0.3)
    else:
        original_clip = concatenate_videoclips(
            [
                ColorClip(
                    (1, 1),
                    color=color,
                )
                .with_duration(0.1)
                .with_fps(fps)
                for color in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            ]
        )
    if with_mask:
        original_clip = original_clip.with_mask(
            ColorClip((1, 1), color=1, is_mask=True).with_fps(fps).with_duration(0.3)
        )

    write_gif_with_imageio(original_clip, filename, fps=fps, logger=None, loop=loop)

    final_clip = VideoFileClip(filename)

    r, g, b = final_clip.get_frame(0)[0][0]
    assert r == 255
    assert g == 0
    assert b == 0

    r, g, b = final_clip.get_frame(0.1)[0][0]
    assert r == 0
    assert g == 255
    assert b == 0

    r, g, b = final_clip.get_frame(0.2)[0][0]
    assert r == 0
    assert g == 0
    assert b == 255


def test_transparent_video(util):
    # Has one R 30%
    clip = ColorClip((100, 100), (255, 0, 0, 76.5)).with_duration(2)
    filename = os.path.join(util.TMP_DIR, "opacity.webm")

    ffmpeg_write_video(clip, filename, codec="libvpx", fps=5)

    # Load output file and check transparency
    result = VideoFileClip(filename, has_mask=True)

    # Check for mask
    assert result.mask is not None

    # Check correct opacity, allow for some tolerance (about 1%)
    # to consider rounding and compressing error
    frame = result.mask.get_frame(1)
    opacity = frame[50, 50]
    assert abs(opacity - 0.3) < 0.01

    result.close()


def test_write_file_with_spaces(util):
    filename = os.path.join(util.TMP_DIR, "name with spaces.mp4")
    clip = ColorClip((1, 1), color=1, is_mask=True).with_fps(1).with_duration(0.3)
    ffmpeg_write_video(clip, filename, fps=1)


def test_write_frame_non_contiguous_matches_contiguous(util):
    """A non-contiguous frame must be written with the same bytes (hence
    layout) as its contiguous equivalent, i.e. row order is preserved.
    """
    size = (32, 32)
    color = (10, 120, 200)
    fps = 5

    contiguous_clip = ColorClip(size, color=color, duration=0.4).with_fps(fps)

    def non_contiguous_frame(t):
        # A strided view over a wider array: not C-contiguous in memory,
        # but the pixels are identical to the contiguous clip's frame.
        wide = np.zeros((size[1], size[0] * 2, 3), dtype=np.uint8)
        wide[:, ::2] = contiguous_clip.get_frame(t)
        frame = wide[:, ::2]
        assert not frame.flags["C_CONTIGUOUS"]
        return frame

    non_contiguous_clip = VideoClip(
        frame_function=non_contiguous_frame, duration=0.4
    ).with_fps(fps)

    contiguous_file = os.path.join(util.TMP_DIR, "writer_contiguous.avi")
    non_contiguous_file = os.path.join(util.TMP_DIR, "writer_non_contiguous.avi")

    # rawvideo keeps the written frames lossless, so the files can be compared
    # byte per byte.
    ffmpeg_write_video(contiguous_clip, contiguous_file, fps, codec="rawvideo")
    ffmpeg_write_video(non_contiguous_clip, non_contiguous_file, fps, codec="rawvideo")

    assert os.path.getsize(contiguous_file) == os.path.getsize(non_contiguous_file)
    with open(contiguous_file, "rb") as f1, open(non_contiguous_file, "rb") as f2:
        assert f1.read() == f2.read()
