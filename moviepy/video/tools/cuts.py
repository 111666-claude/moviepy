"""Contains everything that can help automate the cuts in MoviePy."""

from collections import defaultdict

import numpy as np

from moviepy.decorators import convert_parameter_to_seconds, use_clip_fps_by_default

# Detection modes accepted by ``detect_scenes``.
_SCENE_DETECTION_METHODS = ("luminosity", "histogram")


@use_clip_fps_by_default
@convert_parameter_to_seconds(["start_time"])
def find_video_period(clip, fps=None, start_time=0.3):
    """Find the period of a video based on frames correlation.

    Parameters
    ----------

    clip : moviepy.Clip.Clip
      Clip for which the video period will be computed.

    fps : int, optional
      Number of frames per second used computing the period. Higher values will
      produce more accurate periods, but the execution time will be longer.

    start_time : float, optional
      First timeframe used to calculate the period of the clip.

    Examples
    --------

    .. code:: python

        from moviepy import *
        from moviepy.video.tools.cuts import find_video_period

        clip = VideoFileClip("media/chaplin.mp4").subclipped(0, 1).loop(2)
        round(videotools.find_video_period(clip, fps=80), 6)
        1
    """

    def frame(t):
        return clip.get_frame(t).flatten()

    timings = np.arange(start_time, clip.duration, 1 / fps)[1:]
    ref = frame(0)
    corrs = [np.corrcoef(ref, frame(t))[0, 1] for t in timings]
    return timings[np.argmax(corrs)]


class FramesMatch:
    """Frames match inside a set of frames.

    Parameters
    ----------

    start_time : float
      Starting time.

    end_time : float
      End time.

    min_distance : float
      Lower bound on the distance between the first and last frames

    max_distance : float
      Upper bound on the distance between the first and last frames
    """

    def __init__(self, start_time, end_time, min_distance, max_distance):
        self.start_time = start_time
        self.end_time = end_time
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.time_span = end_time - start_time

    def __str__(self):  # pragma: no cover
        return "(%.04f, %.04f, %.04f, %.04f)" % (
            self.start_time,
            self.end_time,
            self.min_distance,
            self.max_distance,
        )

    def __repr__(self):  # pragma: no cover
        return self.__str__()

    def __iter__(self):  # pragma: no cover
        return iter(
            (self.start_time, self.end_time, self.min_distance, self.max_distance)
        )

    def __eq__(self, other):
        return (
            other.start_time == self.start_time
            and other.end_time == self.end_time
            and other.min_distance == self.min_distance
            and other.max_distance == self.max_distance
        )


class FramesMatches(list):
    """Frames matches inside a set of frames.

    You can instantiate it passing a list of FramesMatch objects or
    using the class methods ``load`` and ``from_clip``.

    Parameters
    ----------

    lst : list
      Iterable of FramesMatch objects.
    """

    def __init__(self, lst):
        list.__init__(self, sorted(lst, key=lambda e: e.max_distance))

    def best(self, n=1, percent=None):
        """Returns a new instance of FramesMatches object or a FramesMatch
        from the current class instance given different conditions.

        By default returns the first FramesMatch that the current instance
        stores.

        Parameters
        ----------

        n : int, optional
          Number of matches to retrieve from the current FramesMatches object.
          Only has effect when ``percent=None``.

        percent : float, optional
          Percent of the current match to retrieve.

        Returns
        -------

        FramesMatch or FramesMatches : If the number of matches to retrieve is
          greater than 1 returns a FramesMatches object, otherwise a
          FramesMatch.

        """
        if percent is not None:
            n = len(self) * percent / 100
        return self[0] if n == 1 else FramesMatches(self[: int(n)])

    def filter(self, condition):
        """Return a FramesMatches object obtained by filtering out the
        FramesMatch which do not satistify a condition.

        Parameters
        ----------

        condition : func
          Function which takes a FrameMatch object as parameter and returns a
          bool.

        Examples
        --------
        .. code:: python

            # Only keep the matches corresponding to (> 1 second) sequences.
            new_matches = matches.filter(lambda match: match.time_span > 1)
        """
        return FramesMatches(filter(condition, self))

    def save(self, filename):
        """Save a FramesMatches object to a file.

        Parameters
        ----------

        filename : str
          Path to the file in which will be dumped the FramesMatches object data.
        """
        np.savetxt(
            filename,
            np.array([np.array(list(e)) for e in self]),
            fmt="%.03f",
            delimiter="\t",
        )

    @staticmethod
    def load(filename):
        """Load a FramesMatches object from a file.

        Parameters
        ----------

        filename : str
          Path to the file to use loading a FramesMatches object.

        Examples
        --------
        >>> matching_frames = FramesMatches.load("somefile")
        """
        arr = np.loadtxt(filename)
        mfs = [FramesMatch(*e) for e in arr]
        return FramesMatches(mfs)

    @staticmethod
    def from_clip(clip, distance_threshold, max_duration, fps=None, logger="bar"):
        """Finds all the frames that look alike in a clip, for instance to make
        a looping GIF.

        Parameters
        ----------

        clip : moviepy.video.VideoClip.VideoClip
          A MoviePy video clip.

        distance_threshold : float
          Distance above which a match is rejected.

        max_duration : float
          Maximal duration (in seconds) between two matching frames.

        fps : int, optional
          Frames per second (default will be ``clip.fps``).

        logger : str, optional
          Either ``"bar"`` for progress bar or ``None`` or any Proglog logger.

        Returns
        -------

        FramesMatches
            All pairs of frames with ``end_time - start_time < max_duration``
            and whose distance is under ``distance_threshold``.

        Examples
        --------

        We find all matching frames in a given video and turn the best match
        with a duration of 1.5 seconds or more into a GIF:

        .. code:: python

            from moviepy import VideoFileClip
            from moviepy.video.tools.cuts import FramesMatches

            clip = VideoFileClip("foo.mp4").resize(width=200)
            matches = FramesMatches.from_clip(
                clip, distance_threshold=10, max_duration=3,  # will take time
            )
            best = matches.filter(lambda m: m.time_span > 1.5).best()
            clip.subclipped(best.start_time, best.end_time).write_gif("foo.gif")
        """
        N_pixels = clip.w * clip.h * 3

        def dot_product(F1, F2):
            return (F1 * F2).sum() / N_pixels

        frame_dict = {}  # will store the frames and their mutual distances

        def distance(t1, t2):
            uv = dot_product(frame_dict[t1]["frame"], frame_dict[t2]["frame"])
            u, v = frame_dict[t1]["|F|sq"], frame_dict[t2]["|F|sq"]
            return np.sqrt(u + v - 2 * uv)

        matching_frames = []  # the final result.

        for t, frame in clip.iter_frames(with_times=True, logger=logger):
            flat_frame = 1.0 * frame.flatten()
            F_norm_sq = dot_product(flat_frame, flat_frame)
            F_norm = np.sqrt(F_norm_sq)

            for t2 in list(frame_dict.keys()):
                # forget old frames, add 't' to the others frames
                # check for early rejections based on differing norms
                if (t - t2) > max_duration:
                    frame_dict.pop(t2)
                else:
                    frame_dict[t2][t] = {
                        "min": abs(frame_dict[t2]["|F|"] - F_norm),
                        "max": frame_dict[t2]["|F|"] + F_norm,
                    }
                    frame_dict[t2][t]["rejected"] = (
                        frame_dict[t2][t]["min"] > distance_threshold
                    )

            t_F = sorted(frame_dict.keys())

            frame_dict[t] = {"frame": flat_frame, "|F|sq": F_norm_sq, "|F|": F_norm}

            for i, t2 in enumerate(t_F):
                # Compare F(t) to all the previous frames

                if frame_dict[t2][t]["rejected"]:
                    continue

                dist = distance(t, t2)
                frame_dict[t2][t]["min"] = frame_dict[t2][t]["max"] = dist
                frame_dict[t2][t]["rejected"] = dist >= distance_threshold

                for t3 in t_F[i + 1 :]:
                    # For all the next times t3, use d(F(t), F(end_time)) to
                    # update the bounds on d(F(t), F(t3)). See if you can
                    # conclude on whether F(t) and F(t3) match.
                    t3t, t2t3 = frame_dict[t3][t], frame_dict[t2][t3]
                    t3t["max"] = min(t3t["max"], dist + t2t3["max"])
                    t3t["min"] = max(t3t["min"], dist - t2t3["max"], t2t3["min"] - dist)

                    if t3t["min"] > distance_threshold:
                        t3t["rejected"] = True

            # Store all the good matches (end_time,t)
            matching_frames += [
                (t1, t, frame_dict[t1][t]["min"], frame_dict[t1][t]["max"])
                for t1 in frame_dict
                if (t1 != t) and not frame_dict[t1][t]["rejected"]
            ]

        return FramesMatches([FramesMatch(*e) for e in matching_frames])

    def select_scenes(
        self, match_threshold, min_time_span, nomatch_threshold=None, time_distance=0
    ):
        """Select the scenes at which a video clip can be reproduced as the
        smoothest possible way, mainly oriented for the creation of GIF images.

        Parameters
        ----------

        match_threshold : float
          Maximum distance possible between frames. The smaller, the
          better-looping the GIFs are.

        min_time_span : float
          Minimum duration for a scene. Only matches with a duration longer
          than the value passed to this parameters will be extracted.

        nomatch_threshold : float, optional
          Minimum distance possible between frames. If is ``None``, then it is
          chosen equal to ``match_threshold``.

        time_distance : float, optional
          Minimum time offset possible between matches.

        Returns
        -------

        FramesMatches : New instance of the class with the selected scenes.

        Examples
        --------

        .. code:: python

            from pprint import pprint
            from moviepy import *
            from moviepy.video.tools.cuts import FramesMatches

            ch_clip = VideoFileClip("media/chaplin.mp4").subclipped(1, 4)
            mirror_and_clip = [ch_clip.with_effects([vfx.TimeMirror()]), ch_clip]
            clip = concatenate_videoclips(mirror_and_clip)

            result = FramesMatches.from_clip(clip, 10, 3).select_scenes(
                1, 2, nomatch_threshold=0,
            )
            print(result)
            # [(1.0000, 4.0000, 0.0000, 0.0000),
            #  (1.1600, 3.8400, 0.0000, 0.0000),
            #  (1.2800, 3.7200, 0.0000, 0.0000),
            #  (1.4000, 3.6000, 0.0000, 0.0000)]
        """
        if nomatch_threshold is None:
            nomatch_threshold = match_threshold

        dict_starts = defaultdict(lambda: [])
        for start, end, min_distance, max_distance in self:
            dict_starts[start].append([end, min_distance, max_distance])

        starts_ends = sorted(dict_starts.items(), key=lambda k: k[0])

        result = []
        min_start = 0
        for start, ends_distances in starts_ends:
            if start < min_start:
                continue

            ends = [end for (end, min_distance, max_distance) in ends_distances]
            great_matches = [
                (end, min_distance, max_distance)
                for (end, min_distance, max_distance) in ends_distances
                if max_distance < match_threshold
            ]

            great_long_matches = [
                (end, min_distance, max_distance)
                for (end, min_distance, max_distance) in great_matches
                if (end - start) > min_time_span
            ]

            if not great_long_matches:
                continue  # No GIF can be made starting at this time

            poor_matches = {
                end
                for (end, min_distance, max_distance) in ends_distances
                if min_distance > nomatch_threshold
            }
            short_matches = {end for end in ends if (end - start) <= 0.6}

            if not poor_matches.intersection(short_matches):
                continue

            end = max(end for (end, min_distance, max_distance) in great_long_matches)
            end, min_distance, max_distance = next(
                e for e in great_long_matches if e[0] == end
            )

            result.append(FramesMatch(start, end, min_distance, max_distance))
            min_start = start + time_distance

        return FramesMatches(result)

    def write_gifs(self, clip, gifs_dir, **kwargs):
        """Extract the matching frames represented by the instance from a clip
        and write them as GIFs in a directory, one GIF for each matching frame.

        Parameters
        ----------

        clip : video.VideoClip.VideoClip
          A video clip whose frames scenes you want to obtain as GIF images.

        gif_dir : str
          Directory in which the GIF images will be written.

        kwargs
          Passed as ``clip.write_gif`` optional arguments.

        Examples
        --------

        .. code:: python

            import os
            from pprint import pprint
            from moviepy import *
            from moviepy.video.tools.cuts import FramesMatches

            ch_clip = VideoFileClip("media/chaplin.mp4").subclipped(1, 4)
            clip = concatenate_videoclips([ch_clip.time_mirror(), ch_clip])

            result = FramesMatches.from_clip(clip, 10, 3).select_scenes(
                1, 2, nomatch_threshold=0,
            )

            os.mkdir("foo")
            result.write_gifs(clip, "foo")
            # MoviePy - Building file foo/00000100_00000400.gif with imageio.
            # MoviePy - Building file foo/00000115_00000384.gif with imageio.
            # MoviePy - Building file foo/00000128_00000372.gif with imageio.
            # MoviePy - Building file foo/00000140_00000360.gif with imageio.
        """
        for start, end, _, _ in self:
            name = "%s/%08d_%08d.gif" % (gifs_dir, 100 * start, 100 * end)
            clip.subclipped(start, end).write_gif(name, **kwargs)


class SceneChange:
    """A scene change detected between frames of a clip.

    Two kinds of scene changes are reported:

    - ``"hard"``: a hard cut, i.e. a single frame whose color content is very
      different from the previous one. ``start_time`` and ``end_time`` are
      equal and hold the time of the first frame of the new scene.
    - ``"gradual"``: a sustained transition (fade in, fade out, dissolve...).
      ``start_time`` is the time of the first frame affected by the
      transition and ``end_time`` the time of the first frame after it.

    Single-frame flashes (a white or otherwise extreme frame followed by a
    return to the previous content) are not scene changes and are never
    reported.

    Parameters
    ----------

    start_time : float
      Starting time of the change, in seconds, aligned with the clip's
      timeline.

    end_time : float
      Ending time of the change, in seconds. Equal to ``start_time`` for a
      hard cut.

    kind : str
      Either ``"hard"`` or ``"gradual"``.

    score : float
      Histogram distance associated with the change. For a hard cut it is the
      distance between the two frames around the cut; for a gradual change it
      is the largest consecutive-frame distance observed during the
      transition.
    """

    def __init__(self, start_time, end_time, kind, score):
        if kind not in ("hard", "gradual"):
            raise ValueError(
                "SceneChange kind must be 'hard' or 'gradual', got %r" % kind
            )
        self.start_time = start_time
        self.end_time = end_time
        self.kind = kind
        self.score = score
        self.time_span = end_time - start_time

    def __str__(self):  # pragma: no cover
        return "SceneChange(%s, %.04f, %.04f, %.04f)" % (
            self.kind,
            self.start_time,
            self.end_time,
            self.score,
        )

    def __repr__(self):  # pragma: no cover
        return self.__str__()

    def __eq__(self, other):
        return (
            isinstance(other, SceneChange)
            and self.kind == other.kind
            and self.start_time == other.start_time
            and self.end_time == other.end_time
            and self.score == other.score
        )


class SceneChanges(list):
    """The scene changes detected in a clip, as a list of ``SceneChange``.

    Parameters
    ----------

    changes : list
      Iterable of ``SceneChange`` objects, ordered chronologically.

    distances : numpy.ndarray, optional
      Consecutive-frame histogram distances computed during detection, one
      per pair of consecutive sampled frames. Kept for inspection and for
      re-running a detection with another threshold.
    """

    def __init__(self, changes, distances=None):
        list.__init__(self, changes)
        self.distances = (
            None if distances is None else np.asarray(distances, dtype=float)
        )

    @property
    def hard_cuts(self):
        """Sub-list containing only the hard cuts."""
        return [c for c in self if c.kind == "hard"]

    @property
    def gradual_changes(self):
        """Sub-list containing only the gradual (fade, dissolve) changes."""
        return [c for c in self if c.kind == "gradual"]


def _frame_histogram(frame, nb_bins):
    """Compute a normalized per-channel histogram of one frame.

    Mask clips, whose frames are two-dimensional arrays, are treated as a
    single channel. Frames are read as uint8 values in [0, 255]; float mask
    frames in [0, 1] are scaled first. The frame is never copied beyond the
    small histogram: only one frame and one histogram are held in memory at a
    time, so long clips can be processed without caching their frames.
    """
    if frame.ndim == 2:
        channels = 1
        pixels = frame.reshape(-1, 1)
    else:
        channels = frame.shape[-1]
        pixels = frame.reshape(-1, channels)
    if not np.issubdtype(pixels.dtype, np.integer):
        pixels = np.clip(pixels * 255.0, 0, 255).astype(np.uint8)
    hist = np.empty(channels * nb_bins, dtype=float)
    bin_width = 256.0 / nb_bins
    for channel in range(channels):
        hist[channel * nb_bins : (channel + 1) * nb_bins] = np.bincount(
            (pixels[:, channel] // bin_width).astype(np.intp),
            minlength=nb_bins,
        )
    hist /= pixels.shape[0]
    return hist


def _histogram_distance(hist1, hist2, nb_bins):
    """Mean per-channel total variation distance between two histograms.

    Returns a value in [0, 2]: 0 means identical color distributions, 2 means
    fully disjoint channels. Equal-luminosity frames with different colors
    therefore stay far apart.
    """
    channels = hist1.shape[0] // nb_bins
    distance = 0.0
    for channel in range(channels):
        start = channel * nb_bins
        end = start + nb_bins
        distance += np.abs(hist1[start:end] - hist2[start:end]).sum()
    return distance / channels


def _detect_histogram_changes(clip, fps, nb_bins, threshold, logger):
    """Detect hard cuts and gradual transitions from streamed histograms.

    A single streaming pass is performed and no frame is ever cached: only
    the histograms of the current scene's anchor frame, the previous frame
    and, while a transition is open, its first and previous frames are kept.

    The histogram distance between two frames lies in [0, 2]. ``threshold``
    is an absolute cutoff on that distance (as opposed to a multiplier of the
    mean), which keeps the detection reliable on clips mostly made of a
    transition, where mean-relative cutoffs are hidden by the transition
    itself. A two-state machine runs around the anchor frame:

    - in the normal state, a consecutive-frame distance above the cutoff, or a
      cumulative drift away from the anchor above it, opens a transition; the
      anchor is left untouched until a new scene is confirmed so that slow
      sub-threshold drift cannot drag it along;
    - in the transition state, the next frame either returns to the anchor
      (a flash, e.g. a one-frame white frame, which is discarded), matches the
      previous frame while being far from the anchor (the content has settled
      on a new scene: a hard cut if it settled immediately, a gradual
      transition such as a fade or dissolve otherwise), or keeps changing (the
      transition continues).
    """
    changes = []
    distances = []
    in_transition = False
    anchor_hist = None  # histogram of the current confirmed scene
    previous_hist = None
    previous_time = None
    # Transition state.
    start_time = None
    first_hist = None
    transition_prev_hist = None
    pending_settle_hist = None
    peak = 0.0

    for time, frame in clip.iter_frames(fps=fps, with_times=True, logger=logger):
        histogram = _frame_histogram(frame, nb_bins)
        if previous_hist is None:
            anchor_hist = histogram
            previous_hist = histogram
            previous_time = time
            continue

        distance = _histogram_distance(previous_hist, histogram, nb_bins)
        distances.append(distance)

        if not in_transition:
            # A transition starts either on an abrupt jump (hard cut, flash)
            # or when the cumulative drift away from the scene's anchor frame
            # exceeds the cutoff (slow fade whose every step stays small).
            if distance > threshold or (
                distance > 0
                and _histogram_distance(histogram, anchor_hist, nb_bins) > threshold
            ):
                in_transition = True
                start_time = time
                first_hist = histogram
                transition_prev_hist = histogram
                pending_settle_hist = None
                peak = distance
        elif _histogram_distance(histogram, anchor_hist, nb_bins) <= threshold:
            # Content is back to the previous scene: a flash, not a cut.
            in_transition = False
        elif pending_settle_hist is not None:
            # The previous frame was the first one of a candidate new scene:
            # require a second matching frame to confirm, so a short flash
            # cannot masquerade as a hard cut.
            if (
                _histogram_distance(histogram, pending_settle_hist, nb_bins)
                <= threshold
            ):
                if (
                    _histogram_distance(first_hist, pending_settle_hist, nb_bins)
                    <= threshold
                ):
                    # The content never changed after the opening jump.
                    kind = "hard"
                    end_time = start_time
                else:
                    # The content moved before stabilizing: fade or dissolve.
                    kind = "gradual"
                    end_time = time
                changes.append(SceneChange(start_time, end_time, kind, peak))
                anchor_hist = pending_settle_hist
                in_transition = False
            else:
                # Not stable yet: keep tracking the transition.
                pending_settle_hist = None
                peak = max(peak, distance)
                transition_prev_hist = histogram
        elif _histogram_distance(histogram, transition_prev_hist, nb_bins) <= threshold:
            # First frame that matches its predecessor: candidate new scene,
            # wait for the next frame to confirm it.
            pending_settle_hist = histogram
        else:
            # Content keeps moving: the transition continues.
            peak = max(peak, distance)
            transition_prev_hist = histogram

        previous_hist = histogram
        previous_time = time

    # A transition still open when the clip ends. A pending candidate counts
    # as confirmed since no further change was observed; otherwise the whole
    # span is reported as a gradual change.
    if in_transition:
        if pending_settle_hist is not None and (
            _histogram_distance(first_hist, pending_settle_hist, nb_bins) <= threshold
        ):
            changes.append(SceneChange(start_time, start_time, "hard", peak))
        else:
            changes.append(SceneChange(start_time, previous_time, "gradual", peak))

    return SceneChanges(changes, distances=np.array(distances))


def _enforce_min_scene_len(boundaries, min_scene_len, end):
    """Drop internal scene boundaries closer than ``min_scene_len``.

    The first boundary (always 0) and the clip end are always kept, so the
    first and last scenes remain complete. Boundaries are processed in order
    and each accepted boundary becomes the new reference.
    """
    if min_scene_len <= 0 or len(boundaries) <= 2:
        return boundaries

    kept = [boundaries[0]]
    for boundary in boundaries[1:-1]:
        if boundary - kept[-1] >= min_scene_len:
            kept.append(boundary)
    if kept[-1] != end:
        kept.append(end)
    return kept


def _resolve_fps(clip, fps):
    """Return an explicit fps, falling back to the clip's own fps."""
    if fps is not None:
        return fps
    if clip is not None and getattr(clip, "fps", None):
        return clip.fps
    raise AttributeError(
        "No 'fps' (frames per second) attribute specified for detect_scenes"
        " and the clip has no 'fps' attribute. Either provide e.g. fps=24 in"
        " the arguments, or define the clip's fps with `clip.fps = 24`"
    )


def detect_scenes(
    clip=None,
    luminosities=None,
    luminosity_threshold=10,
    logger="bar",
    fps=None,
    method="luminosity",
    histogram_threshold=0.5,
    min_scene_len=0,
    nb_bins=32,
):
    """Detects scenes of a clip based on luminosity or color changes.

    Two detection methods are available:

    - ``"luminosity"`` (default, historical behaviour): scenes are split on
      large jumps of the total frame luminosity. It cannot distinguish two
      equally luminous shots of different colors.
    - ``"histogram"``: scenes are split on changes of the per-channel RGB
      color histogram distribution. Hard cuts between equally luminous but
      differently colored shots are detected, sustained transitions (fades,
      dissolves) are reported with their start and end times, and single
      frame flashes such as a one-frame white flash are ignored rather than
      treated as scenes.

    Note that for large clips this may take some time. Frames are always
    streamed one at a time and are never cached, so the memory usage does not
    grow with the clip length.

    Returns
    -------

    tuple : (cuts, data)
      ``cuts`` is a series of cuts ``[(0, t1), (t1, t2), ..., (tn, end)]``
      covering the whole clip, so the first and last scenes are always
      complete. ``data`` is the array of per-frame luminosities when
      ``method="luminosity"`` and a :class:`SceneChanges` object when
      ``method="histogram"``.

    Parameters
    ----------

    clip : video.VideoClip.VideoClip, optional
      A video clip. Can be None if a list of luminosities is provided instead
      (luminosity method only). If provided, the frames of the clip will be
      streamed. If the clip has no 'fps' attribute, you must provide ``fps``.

    luminosities : list, optional
      A list of luminosities, e.g. returned by detect_scenes in a previous
      run. Only usable with ``method="luminosity"``.

    luminosity_threshold : float, optional
      Determines a threshold above which the 'luminosity jumps' will be
      considered as scene changes. A scene change is defined as a change
      between 2 consecutive frames that is larger than ``avg * thr`` where
      avg is the average of the absolute changes between consecutive frames.
      Only used with ``method="luminosity"``.

    logger : str, optional
      Either ``"bar"`` for progress bar or ``None`` or any Proglog logger.

    fps : int, optional
      Frames per second value. Must be provided if you provide no clip or a
      clip without fps attribute.

    method : str, optional
      Either ``"luminosity"`` (default) or ``"histogram"``.

    histogram_threshold : float, optional
      Absolute cutoff on the consecutive-frame color histogram distance, which
      ranges from 0 (identical color distributions) to 2 (fully disjoint
      channels). Frames farther apart than this value are considered
      different. Lower values detect subtler changes but produce more false
      positives. Only used with ``method="histogram"``. Must be strictly
      positive.

    min_scene_len : float, optional
      Minimum scene length in seconds. Detected boundaries closer than this to
      the previous accepted boundary are merged away. The boundaries at 0 and
      at the clip end are always kept. Defaults to 0, which keeps every
      detected boundary.

    nb_bins : int, optional
      Number of bins per color channel used to build the histograms. Must be
      between 2 and 256 (a single bin would make every histogram identical).
      Only used with ``method="histogram"``.
    """
    if method not in _SCENE_DETECTION_METHODS:
        raise ValueError(
            "method must be one of %s, got %r"
            % (", ".join(repr(m) for m in _SCENE_DETECTION_METHODS), method)
        )
    if fps is not None and fps <= 0:
        raise ValueError("fps must be strictly positive, got %r" % fps)
    if min_scene_len < 0:
        raise ValueError("min_scene_len must be non-negative, got %r" % min_scene_len)

    if method == "histogram":
        if clip is None:
            raise ValueError("method='histogram' requires a clip, got clip=None")
        if histogram_threshold <= 0:
            raise ValueError(
                "histogram_threshold must be strictly positive, got %r"
                % histogram_threshold
            )
        if (
            isinstance(nb_bins, bool)
            or not isinstance(nb_bins, (int, np.integer))
            or not (2 <= nb_bins <= 256)
        ):
            raise ValueError(
                "nb_bins must be an integer between 2 and 256, got %r" % nb_bins
            )

        fps = _resolve_fps(clip, fps)
        end = clip.duration
        if end is None or end <= 0:
            return [(0, 0)], SceneChanges([], distances=np.array([]))

        scene_changes = _detect_histogram_changes(
            clip, fps, nb_bins, histogram_threshold, logger
        )

        boundaries = [0.0]
        for change in scene_changes:
            if change.kind == "hard":
                boundary = change.start_time
            else:
                # End of the gradual transition: first frame of the new scene.
                boundary = change.end_time
            if boundary > 0 and boundary < end and boundary != boundaries[-1]:
                boundaries.append(boundary)
        boundaries.append(end)
        boundaries = _enforce_min_scene_len(boundaries, min_scene_len, end)
        cuts = [(t1, t2) for t1, t2 in zip(boundaries, boundaries[1:])]
        return cuts, scene_changes

    if luminosities is None and clip is None:
        raise ValueError("detect_scenes needs either a clip or a list of luminosities")
    if fps is None and clip is not None:
        fps = _resolve_fps(clip, fps)
    elif fps is None:
        raise AttributeError(
            "No 'fps' (frames per second) attribute specified for"
            " detect_scenes and no clip was provided. Provide e.g. fps=24 in"
            " the arguments, or pass a clip with an 'fps' attribute"
        )

    if luminosities is None:
        luminosities = [
            f.sum() for f in clip.iter_frames(fps=fps, dtype="uint32", logger=logger)
        ]

    luminosities = np.array(luminosities, dtype=float)
    if clip is not None:
        end = clip.duration
    else:
        end = len(luminosities) * (1.0 / fps)
    luminosity_diffs = abs(np.diff(luminosities))
    if luminosity_diffs.size:
        avg = luminosity_diffs.mean()
        luminosity_jumps = (
            1 + np.array(np.nonzero(luminosity_diffs > luminosity_threshold * avg))[0]
        )
    else:
        luminosity_jumps = np.array([], dtype=int)
    timings = [0] + list((1.0 / fps) * luminosity_jumps) + [end]
    timings = _enforce_min_scene_len(timings, min_scene_len, end)
    cuts = [(t1, t2) for t1, t2 in zip(timings, timings[1:])]
    return cuts, luminosities
