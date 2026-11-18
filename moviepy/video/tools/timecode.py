"""SMPTE timecode helpers used by the ``Timecode`` video effect.

This module only deals with the frame/timecode arithmetic. It supports
integer frame rates (for instance ``24``, ``25`` or ``30``) as non
drop-frame timecodes, as well as the two standard drop-frame frame rates
``29.97`` (30000/1001) and ``59.94`` (60000/1001).

All frame numbers are *absolute* zero-based frame counts. A timecode is a
``HH:MM:SS:FF`` string. For drop-frame timecodes the separator between
seconds and frames may also be a semicolon (``HH:MM:SS;FF``) following the
SMPTE convention; both separators are accepted when parsing.
"""

from fractions import Fraction

# Standard drop-frame frame rates, expressed as exact fractions.
_DF_RATES = {
    Fraction(30000, 1001): {"rate": 30, "frames_per_10min": 17982},
    Fraction(60000, 1001): {"rate": 60, "frames_per_10min": 35964},
}

# Number of frames skipped at each drop-frame minute boundary (per minute).
# 2 frames at 29.97fps, 4 frames at 59.94fps.
_DF_DROP = {30: 2, 60: 4}


def _is_integer(value):
    """Return True for genuine integers (booleans are rejected)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _fps_as_fraction(fps):
    """Normalize a frame rate provided as int, float, Fraction or str.

    Returns a ``Fraction`` along with the drop-frame configuration when the
    rate is one of the supported drop-frame rates.
    """
    if (
        fps is None
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float, Fraction, str))
    ):
        raise TypeError(
            "fps must be a number or a string, got {} ({})".format(
                fps, type(fps).__name__
            )
        )

    if isinstance(fps, str):
        try:
            fps_fraction = Fraction(fps)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("Invalid frame rate {!r}".format(fps)) from exc
    else:
        fps_fraction = Fraction(fps)

    if fps_fraction <= 0:
        raise ValueError("fps must be strictly positive, got {}".format(fps))

    return fps_fraction


def _drop_frame_config(fps, drop_frame):
    """Validate the fps / drop-frame combination.

    Returns a tuple ``(rate, frames_per_10min, drop_count)`` for drop-frame
    timecodes, or ``None`` for non drop-frame timecodes.
    """
    fps_fraction = _fps_as_fraction(fps)

    if drop_frame:
        # Only 29.97 and 59.94 have a defined drop-frame behavior. Compare
        # with a small tolerance so that float literals (29.97) and exact
        # fractions (30000/1001) are treated as the same rate.
        for rate_fraction, config in _DF_RATES.items():
            if abs(float(fps_fraction) - float(rate_fraction)) < 0.005:
                rate = config["rate"]
                return (
                    rate,
                    config["frames_per_10min"],
                    _DF_DROP[rate],
                )
        raise ValueError(
            "Drop-frame timecodes are only defined for 29.97 and 59.94 fps, "
            "got {} fps".format(fps)
        )
    return None


def frame_to_timecode(frame_number, fps, drop_frame=False):
    """Convert an absolute frame number into a ``HH:MM:SS:FF`` timecode.

    Parameters
    ----------

    frame_number : int
      Zero-based absolute frame count. Must be a non-negative integer.

    fps : int, float, Fraction or str
      Frame rate. Integer rates produce non drop-frame timecodes while
      ``29.97`` and ``59.94`` (or their exact fractional equivalents
      ``30000/1001`` and ``60000/1001``) can be used with ``drop_frame=True``.

    drop_frame : bool, optional
      Whether to compute a drop-frame timecode. Only valid for 29.97 and
      59.94 fps.
    """
    if not _is_integer(frame_number):
        raise TypeError(
            "frame_number must be an integer, got {} ({})".format(
                frame_number, type(frame_number).__name__
            )
        )
    if frame_number < 0:
        raise ValueError(
            "frame_number must be non-negative, got {}".format(frame_number)
        )

    df_config = _drop_frame_config(fps, drop_frame)

    if df_config is None:
        # Non drop-frame: the frame rate must be an integer number of frames
        # per second so that a HH:MM:SS:FF label is well defined.
        fps_fraction = _fps_as_fraction(fps)
        if fps_fraction.denominator != 1:
            raise ValueError(
                "Non drop-frame timecodes require an integer frame rate, "
                "got {} fps".format(fps)
            )
        rate = fps_fraction.numerator
        total_frames = frame_number
    else:
        rate, frames_per_10min, drop_count = df_config
        frames_per_minute = rate * 60
        frames_per_short_minute = frames_per_minute - drop_count

        # Frames elapsed in complete 10-minute chunks and within the chunk.
        chunks = frame_number // frames_per_10min
        remainder = frame_number % frames_per_10min

        # Count the drop-frame minute boundaries already crossed inside this
        # chunk. The first minute is full (1800/3600 frames); every following
        # minute is short by ``drop_count`` frames. The closed form
        # ``1 + (r - full) // short`` is clamped to zero for the first minute.
        crossed = max(
            0,
            1 + (remainder - frames_per_minute) // frames_per_short_minute,
        )
        if crossed > 9:
            crossed = 9
        relabeled = remainder + drop_count * crossed

        minutes = 10 * chunks + relabeled // frames_per_minute
        frames_in_minute = relabeled % frames_per_minute

        total_frames = minutes * frames_per_minute + frames_in_minute

    frames_per_hour = rate * 3600
    if total_frames >= 24 * frames_per_hour:
        raise ValueError(
            "Frame number {} exceeds 24 hours at {} fps".format(frame_number, fps)
        )

    hours, remainder = divmod(total_frames, frames_per_hour)
    minutes, remainder = divmod(remainder, rate * 60)
    seconds, frames = divmod(remainder, rate)

    separator = ";" if drop_frame else ":"
    return "{:02d}:{:02d}:{:02d}{}{:02d}".format(
        hours, minutes, seconds, separator, frames
    )


def timecode_to_frame(timecode, fps, drop_frame=None):
    """Parse a ``HH:MM:SS:FF`` timecode into an absolute frame number.

    Parameters
    ----------

    timecode : str
      Timecode to parse. Both ``:`` and ``;`` separators are accepted; a
      semicolon before the frame field denotes a drop-frame timecode.

    fps : int, float, Fraction or str
      Frame rate of the timecode.

    drop_frame : bool, optional
      Force the drop-frame mode. When ``None`` (default), the mode is inferred
      from the timecode separator (``;`` means drop-frame). An explicit value
      must be consistent with the separator and with the frame rate.
    """
    if not isinstance(timecode, str):
        raise TypeError(
            "timecode must be a string, got {} ({})".format(
                timecode, type(timecode).__name__
            )
        )

    normalized = timecode.strip().replace(";", ":")
    parts = normalized.split(":")
    if len(parts) != 4 or any(part == "" for part in parts):
        raise ValueError(
            "Timecode {!r} must have the HH:MM:SS:FF format".format(timecode)
        )
    try:
        hours, minutes, seconds, frames = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            "Timecode {!r} must contain integer fields".format(timecode)
        ) from exc

    inferred_drop = ";" in timecode
    if drop_frame is None:
        drop_frame = inferred_drop
    elif inferred_drop and not drop_frame:
        raise ValueError(
            "Timecode {!r} uses a drop-frame separator but drop_frame=False".format(
                timecode
            )
        )

    df_config = _drop_frame_config(fps, drop_frame)

    if df_config is None:
        fps_fraction = _fps_as_fraction(fps)
        if fps_fraction.denominator != 1:
            raise ValueError(
                "Non drop-frame timecodes require an integer frame rate, "
                "got {} fps".format(fps)
            )
        rate = fps_fraction.numerator
    else:
        rate, frames_per_10min, drop_count = df_config

    if hours < 0 or minutes < 0 or seconds < 0 or frames < 0:
        raise ValueError("Timecode {!r} contains negative fields".format(timecode))
    if minutes >= 60 or seconds >= 60 or frames >= rate:
        raise ValueError(
            "Timecode {!r} is out of range for {} fps".format(timecode, rate)
        )
    if drop_frame and minutes % 10 != 0 and seconds == 0 and frames < drop_count:
        raise ValueError(
            "Timecode {!r} is not a valid drop-frame label (frames {} to {} "
            "are skipped at this minute boundary)".format(timecode, 0, drop_count - 1)
        )
    if hours >= 24:
        raise ValueError("Timecode {!r} exceeds 24 hours".format(timecode))

    label_frames = hours * rate * 3600 + minutes * rate * 60 + seconds * rate + frames

    if not drop_frame:
        return label_frames

    # Undo the drop-frame relabeling: remove the frames skipped at every
    # minute boundary except every tenth minute.
    total_minutes = hours * 60 + minutes
    skipped = drop_count * (total_minutes - total_minutes // 10)
    return label_frames - skipped
