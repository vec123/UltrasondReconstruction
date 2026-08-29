"""Subject-id based file matching, shared by every stage that pairs files
(shape VTPs, landmark CSVs/VTPs, transforms, ...) coming from different
directories by subject id."""

import os


def subject_id_from_path(path, strip_suffixes=()):
    """
    Reduces a file name to its bare subject id, so that files written by
    different stages still pair up, e.g. '1-01_landmarks.vtp' -> '1-01'.
    """
    stem = os.path.splitext(os.path.basename(path))[0]

    for suffix in strip_suffixes:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    return stem


def match_by_subject_id(file_lists, strip_suffixes=()):
    """
    Groups several lists of file paths by subject id.

    Parameters
    ----------
    file_lists : sequence of lists of file paths, e.g. [csv_files, vtp_files].
    strip_suffixes : suffixes stripped from every file stem before matching,
        e.g. ("_landmarks",).

    Returns
    -------
    common_ids : sorted list of subject ids present in every list.
    id_maps : list of {subject_id: path} dicts, one per input list.
    """
    id_maps = [
        {subject_id_from_path(f, strip_suffixes): f for f in files}
        for files in file_lists
    ]
    common_ids = sorted(set.intersection(*(set(m) for m in id_maps))) if id_maps else []
    return common_ids, id_maps
