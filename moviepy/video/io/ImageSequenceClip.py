"""Implements ImageSequenceClip, a class to create a video clip from a set
of image files.
"""

import os

import numpy as np
import proglog
from imageio.v2 import imread

from moviepy.Clip import Clip
from moviepy.tools import find_extension
from moviepy.video.io.ffmpeg_writer import ffmpeg_write_video_vfr
from moviepy.video.VideoClip import VideoClip


class ImageSequenceClip(VideoClip):
    """A VideoClip made from a series of images.

    Parameters
    ----------

    sequence
      Can be one of these:

      - The name of a folder (containing only pictures). The pictures
        will be considered in alphanumerical order.
      - A list of names of image files. In this case you can choose to
        load the pictures in memory pictures
      - A list of Numpy arrays representing images.

    fps
      Number of picture frames to read per second. Instead, you can provide
      the duration of each image with durations (see below)

    durations
      List of the duration of each picture.

    with_mask
      Should the alpha layer of PNG images be considered as a mask ?

    is_mask
      Will this sequence of pictures be used as an animated mask.

    load_images
      Specify that all images should be loaded into the RAM. This is only
      interesting if you have a small number of images that will be used
      more than once.
    """

    def __init__(
        self,
        sequence,
        fps=None,
        durations=None,
        with_mask=True,
        is_mask=False,
        load_images=False,
    ):
        # CODE WRITTEN AS IT CAME, MAY BE IMPROVED IN THE FUTURE

        if (fps is None) and (durations is None):
            raise ValueError("Please provide either 'fps' or 'durations'.")
        VideoClip.__init__(self, is_mask=is_mask)

        # Parse the data

        self.fromfiles = True

        if isinstance(sequence, list):
            if isinstance(sequence[0], str):
                if load_images:
                    sequence = [imread(file) for file in sequence]
                    self.fromfiles = False
            else:
                # sequence is already a list of numpy arrays
                self.fromfiles = False
        else:
            # sequence is a folder name, make it a list of files:
            sequence = sorted(
                [os.path.join(sequence, file) for file in os.listdir(sequence)]
            )

        # check that all the images are of the same size
        if isinstance(sequence[0], str):
            size = imread(sequence[0]).shape
        else:
            size = sequence[0].shape

        for image in sequence:
            image1 = image
            if isinstance(image, str):
                image1 = imread(image)
            if size != image1.shape:
                raise Exception(
                    "MoviePy: ImageSequenceClip requires all images to be the same size"
                )

        self.fps = fps
        if fps is not None:
            durations = [1.0 / fps for image in sequence]
            self.images_starts = [
                1.0 * i / fps - np.finfo(np.float32).eps for i in range(len(sequence))
            ]
        else:
            self.images_starts = [0] + list(np.cumsum(durations))
        self.durations = durations
        self.duration = sum(durations)
        self.end = self.duration
        self.sequence = sequence

        if fps is None:
            self.fps = len(sequence) / self.duration

        if self.fromfiles:
            self.last_index = None
            self.last_image = None

            if with_mask and (imread(self.sequence[0]).shape[2] == 4):
                self.mask = ImageSequenceClip(
                    sequence=sequence,
                    fps=fps,
                    durations=durations,
                    with_mask=False,
                    is_mask=True,
                    load_images=load_images,
                )

        else:
            if with_mask and (self.sequence[0].shape[2] == 4):
                self.mask = ImageSequenceClip(
                    sequence=sequence,
                    fps=fps,
                    durations=durations,
                    with_mask=False,
                    is_mask=True,
                    load_images=load_images,
                )

        self.size = self.frame_function(0).shape[:2][::-1]

    def write_videofile_vfr(
        self,
        filename,
        codec="libx264",
        bitrate=None,
        audio=True,
        audio_fps=44100,
        preset="medium",
        audio_nbytes=4,
        audio_codec=None,
        audio_bitrate=None,
        audio_bufsize=2000,
        temp_audiofile=None,
        temp_audiofile_path="",
        remove_temp=True,
        write_logfile=False,
        threads=None,
        ffmpeg_params=None,
        logger="bar",
        pixel_format=None,
    ):
        """Write the clip to a variable frame rate (VFR) video file.

        Each image of the sequence is encoded exactly once and stays on
        screen for its entry in ``durations``, instead of being resampled
        at a fixed frame rate like ``write_videofile`` does. This avoids
        generating large amounts of identical frames for long still images.

        The output container must support VFR timestamps: use a ``.mp4``
        or ``.mkv`` filename.

        Parameters
        ----------

        filename
          Name of the video file to write, with extension ``.mp4`` or
          ``.mkv``.

        codec
          Codec to use for image encoding, ``'libx264'`` by default.

        audio
          Either ``True``, ``False``, or a file name. If ``True`` and the
          clip has an audio clip attached, this audio clip will be
          incorporated as a soundtrack in the movie, synchronized with the
          variable frame rate video. If ``audio`` is the name of an audio
          file, this file is incorporated instead.

        The other parameters are the same as in
        ``VideoClip.write_videofile``, except that there is no ``fps``
        parameter: the timing is entirely driven by the clip's durations.

        Examples
        --------

        .. code:: python

            clip = ImageSequenceClip(images, durations=[0.2, 1.0, 0.4])
            clip.write_videofile_vfr("slideshow.mp4")
        """
        name, ext = os.path.splitext(os.path.basename(filename))
        ext = ext[1:].lower()
        logger = proglog.default_bar_logger(logger)

        if ext not in ("mp4", "mkv"):
            raise ValueError(
                "MoviePy error: write_videofile_vfr requires a container "
                "supporting variable frame rate timestamps: '.mp4' or "
                f"'.mkv' (got '{os.path.basename(filename)}')."
            )

        durations = self.durations
        if len(durations) != len(self.sequence):
            raise ValueError(
                "MoviePy error: the number of durations "
                f"({len(durations)}) does not match the number of images "
                f"({len(self.sequence)})."
            )
        for d in durations:
            if not np.isfinite(d) or d <= 0:
                raise ValueError(
                    "MoviePy error: write_videofile_vfr requires every "
                    "image duration to be finite and strictly positive, "
                    f"got {d}."
                )

        if audio_codec is None:
            # libvorbis is preferred for Matroska: unlike mp3, its
            # timestamps carry no encoder delay, which keeps the
            # soundtrack synchronized with the VFR video frames.
            audio_codec = "libvorbis" if ext == "mkv" else "libmp3lame"
        elif audio_codec == "raw16":
            audio_codec = "pcm_s16le"
        elif audio_codec == "raw32":
            audio_codec = "pcm_s32le"

        audiofile = audio if isinstance(audio, str) else None
        make_audio = (
            (audiofile is None) and (audio is True) and (self.audio is not None)
        )

        if make_audio and temp_audiofile:
            # The audio will be the clip's audio
            audiofile = temp_audiofile
        elif make_audio:
            audio_ext = find_extension(audio_codec)
            audiofile = os.path.join(
                temp_audiofile_path,
                name + Clip._TEMP_FILES_PREFIX + "wvf_snd.%s" % audio_ext,
            )

        logger(message="MoviePy - Building video %s." % filename)
        if make_audio:
            self.audio.write_audiofile(
                audiofile,
                audio_fps,
                audio_nbytes,
                audio_bufsize,
                audio_codec,
                bitrate=audio_bitrate,
                write_logfile=write_logfile,
                logger=logger,
            )
            # The audio is already encoded,
            # so there is no need to encode it during video export
            audio_codec = "copy"

        try:
            ffmpeg_write_video_vfr(
                self,
                filename,
                durations,
                codec=codec,
                bitrate=bitrate,
                preset=preset,
                write_logfile=write_logfile,
                audiofile=audiofile,
                audio_codec=audio_codec,
                threads=threads,
                ffmpeg_params=ffmpeg_params,
                logger=logger,
                pixel_format=pixel_format,
            )
        finally:
            if remove_temp and make_audio:
                if os.path.exists(audiofile):
                    os.remove(audiofile)
        logger(message="MoviePy - video ready %s" % filename)

    def _find_image_index(self, t):
        return max([i for i in range(len(self.sequence)) if self.images_starts[i] <= t])

    def frame_function(self, t):
        """Retrieves the frame corresponding to the given time `t`.

        Depending on whether the frames are loaded from files or provided as
        an in-memory sequence, this function either reads the frame from disk
        or accesses it directly from the sequence.

        Parameters
        ----------

        t (float): The time (in seconds) for which the corresponding frame
            is to be retrieved.

        Returns
        -------

        numpy.ndarray: The image frame at the specified time. If `is_mask`
            is True, the alpha channel of the image is returned
            as a float array normalized to the range [0, 1].
            Otherwise, the RGB channels of the image are returned.

        """
        index = self._find_image_index(t)

        if self.fromfiles:
            self.last_index = index

            if self.is_mask:
                self.last_image = (
                    imread(self.sequence[index])[:, :, 3].astype(float) / 255.0
                )
            else:
                self.last_image = imread(self.sequence[index])[:, :, :3]

            return self.last_image
        else:
            if self.is_mask:
                return self.sequence[index][:, :, 3].astype(float) / 255.0
            else:
                return self.sequence[index][:, :, :3]
