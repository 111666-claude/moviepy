"""Cutting utilities working with audio."""

import numpy as np

from moviepy.decorators import requires_duration, requires_fps


def find_audio_period(clip, min_time=0.1, max_time=2, time_resolution=0.01):
    """Finds the period, in seconds of an audioclip.

    Parameters
    ----------

    min_time : float, optional
      Minimum bound for the returned value.

    max_time : float, optional
      Maximum bound for the returned value.

    time_resolution : float, optional
      Numerical precision.
    """
    chunksize = int(time_resolution * clip.fps)
    chunk_duration = 1.0 * chunksize / clip.fps
    # v denotes the list of volumes
    v = np.array([(chunk**2).sum() for chunk in clip.iter_chunks(chunksize)])
    v = v - v.mean()
    corrs = np.correlate(v, v, mode="full")[-len(v) :]
    corrs[: int(min_time / chunk_duration)] = 0
    corrs[int(max_time / chunk_duration) :] = 0
    return chunk_duration * np.argmax(corrs)


@requires_duration
@requires_fps
def find_audio_onsets(
    clip,
    window_duration=0.05,
    energy_threshold=1.5,
    min_onset_distance=0.2,
    max_duration=None,
):
    """Detects the onset (attack) times of an audio clip from its energy.

    The audio is read window by window and the energy of every window is
    computed. A window is reported as an onset when its energy jumps above the
    median energy of the preceding windows (by a factor ``energy_threshold``)
    and is a local maximum. This selects the "attacks" of the signal (e.g. the
    drum hits of a music track), which are stable candidates for cutting a
    video on the beat.

    The audio is streamed through ``AudioClip.iter_chunks``, one window at
    a time, so a long track is never fully loaded into memory. Reading stops as
    soon as ``max_duration`` is reached when provided.

    Parameters
    ----------

    clip : moviepy.audio.AudioClip.AudioClip
      Audio clip in which the onsets will be detected. It must have a
      ``duration`` and an ``fps`` set.

    window_duration : float, optional
      Length, in seconds, of the analysis window. A track shorter than one
      window cannot contain any detectable onset.

    energy_threshold : float, optional
      Minimum ratio between the energy of an onset window and the median
      energy of the preceding windows for the onset to be retained. The larger
      the value, the more selective (and stable) the detection.

    min_onset_distance : float, optional
      Minimum time, in seconds, separating two reported onsets. When several
      onsets are closer than this, only the strongest one is kept.

    max_duration : float, optional
      Only analyze the audio up to this time, in seconds. When ``None`` the
      whole clip is analyzed.

    Returns
    -------

    list
      A list of onset times in seconds, strictly increasing and always starting
      with ``0`` (the beginning of the track is always a valid cut point).

    Examples
    --------

    .. code:: python

        from moviepy.audio.tools.cuts import find_audio_onsets

        onsets = find_audio_onsets(music, window_duration=0.05)
        # [0, 0.5, 1.0, 1.5, ...]
    """
    if window_duration is None or window_duration <= 0:
        raise ValueError("'window_duration' must be a strictly positive number")
    if energy_threshold <= 0:
        raise ValueError("'energy_threshold' must be a strictly positive number")
    if min_onset_distance is None or min_onset_distance < 0:
        raise ValueError("'min_onset_distance' must be a positive number or zero")
    if max_duration is not None and max_duration <= 0:
        raise ValueError("'max_duration' must be a strictly positive number")

    duration = (
        clip.duration if max_duration is None else min(clip.duration, max_duration)
    )

    # A single window, or less, cannot contain any energy change: the only
    # stable cut point is the beginning of the track.
    if duration < window_duration:
        return [0.0]

    chunksize = max(1, int(window_duration * clip.fps))
    energies = []
    times = []
    current_time = 0.0

    # ``iter_chunks`` is a generator: breaking out of the loop stops the
    # decoding, keeping the memory footprint bounded to one window.
    for chunk in clip.iter_chunks(chunksize, fps=clip.fps, logger=None):
        if current_time >= duration:
            break
        energies.append(float(np.mean(chunk**2)))
        times.append(current_time)
        current_time += len(chunk) / clip.fps

    energies = np.array(energies, dtype=float)
    times = np.array(times, dtype=float)

    # A silent or constant-energy track has no attack, only the first frame is
    # a meaningful cut point.
    if not np.any(energies > 0):
        return [0.0]

    # The ratio test alone is unstable while the first windows are silent: the
    # median baseline is then ~0 and any tiny fluctuation looks like an onset.
    # Requiring a minimum absolute energy, expressed as a fraction of the peak
    # energy of the analyzed track, removes these spurious detections.
    min_onset_energy = 0.05 * energies.max()

    onset_indices = []
    for index in range(1, len(energies)):
        baseline = np.median(energies[:index])
        strength = energies[index] / baseline if baseline > 0 else np.inf
        is_local_max = energies[index] >= energies[index - 1] and (
            index == len(energies) - 1 or energies[index] >= energies[index + 1]
        )
        if (
            strength >= energy_threshold
            and energies[index] >= min_onset_energy
            and is_local_max
        ):
            onset_indices.append(index)

    # Enforce the minimum spacing, keeping the strongest onset of each cluster.
    selected_indices = []
    for index in onset_indices:
        if selected_indices and (
            times[index] - times[selected_indices[-1]] < min_onset_distance
        ):
            if energies[index] > energies[selected_indices[-1]]:
                selected_indices[-1] = index
            continue
        selected_indices.append(index)

    # ``0`` is always returned and always comes first: the beginning of the
    # track is a valid, zero-cost cut point. Onset indices are always >= 1
    # because the first window has no history to compare against.
    return [0.0] + [float(times[index]) for index in selected_indices]
