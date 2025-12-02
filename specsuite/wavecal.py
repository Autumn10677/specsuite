import numpy as np
from tqdm import tqdm
from collections import defaultdict


def build_hash_table(
    lines: np.ndarray,
    rounding: int = 3,
    min_separation: float = 0.0,
    max_separation: float = np.inf,
) -> dict:
    """
    Converts a 1D line list into a hash table where
    keys represent the location of a point in a
    two-point reference frame. Specifically, two
    points form a 'base pair' that is used to
    determine a relative scaling. So, if you had...

        'base pair': (100, 200)
        'point': 150

    ..the relative 'location' of the above point
    would be 0.5.

    Parameters:
    -----------
    lines :: np.ndarray
        A 1D list containing the locations of line
        emissions. These can be either floats or
        integers.
    rounding :: int
        How many decimals to keep when calculating
        the relative distances of points.
    max_separation :: float
        The maximum allowed separation between base
        pair points. If no argument is provided,
        this will default to 'np.inf'.

    Returns:
    --------
    hash_table :: dict
        A hash table where relative point locations
        are used for the keys, and a list of dictionaries
        containing base pairs / points are used for the
        values.
    """

    # Ensures that lines are the lines are ordered and unique
    lines = np.sort(np.unique(lines))
    N = len(lines)

    # Using a 'defaultdict' allows us to append to values of new keys
    hash_table = defaultdict(list)

    # Iterates over every point (will be the first base point)
    for i in tqdm(range(N)):

        # Determines which base pairs are valid (finding valid second base points)
        scales = lines - lines[i]
        valid_scale_mask = (
            (np.abs(scales) > min_separation)
            & (np.abs(scales) < max_separation)
            & (scales != 0)
        )
        valid_scale_indices = np.where(valid_scale_mask)[0]

        # Iterates over all valid base pair combinations
        for k in valid_scale_indices:

            scale = scales[k]
            distances = (lines - lines[i]) / scale

            # Mask out points that are too close / far from the base pair origin
            valid_mask = (distances != 0) & (np.abs(distances) < 1.0)
            if not np.any(valid_mask):
                continue

            # Calculates the rounded distances to form hash keys
            rel_lines = lines[valid_mask]
            rel_distances = np.round(distances[valid_mask], rounding)
            hash_keys = np.round(rel_distances, rounding)

            # Append all results to the hash table
            for key, point in zip(hash_keys, rel_lines):
                hash_table[key].append(
                    {
                        "base pair": (lines[i], lines[k]),
                        "point": point,
                    }
                )

    return hash_table


def cast_votes(
    hash_table: dict,
    lines: np.ndarray,
    rounding: int = 3,
    sigma: float = 3,
) -> dict:
    """
    Looks for lines that have a relative spacing described
    in the provided hash table. If the relative position of
    a point is found in the hash table, every point listed
    at that entry recieves one 'vote'.

    Parameters:
    -----------
    hash_table :: dict

    lines :: np.ndarray

    rounding :: int

    Returns:
    --------
    votes :: dict
    """

    # Ensures that lines are the lines are ordered and unique
    lines = np.sort(np.unique(lines))
    N = len(lines)

    # Using a 'defaultdict' allows us to append to values of new keys
    votes = defaultdict(int)
    hash_keys_set = set(hash_table.keys())

    # Iterates over every point (will be the first base point)
    for i in tqdm(range(N)):

        # Determines which base pairs are valid (finding valid second base points)
        scales = lines - lines[i]
        valid_scale_mask = scales != 0
        valid_scale_indices = np.where(valid_scale_mask)[0]

        # Iterates over all valid base pair combinations
        for k in valid_scale_indices:

            scale = scales[k]
            distances = (lines - lines[i]) / scale

            # Mask out points that are too close / far from the base pair origin
            valid_mask = (distances != 0) & (np.abs(distances) < 1.0)
            if not np.any(valid_mask):
                continue

            # Calculates the rounded distances to form hash keys
            rel_distances = np.round(distances[valid_mask], rounding)
            hash_keys = (rel_distances * 10**rounding).astype(int)

            mask_in_table = np.isin(hash_keys, list(hash_keys_set))
            if not np.any(mask_in_table):
                continue

            for key in hash_keys[mask_in_table]:
                for pd in hash_table[key]:
                    votes[pd["base pair"][0]] += 1
                    votes[pd["base pair"][1]] += 1
                    votes[pd["point"]] += 1

    return votes


def get_most_likely_features(
    votes,
    keep_top_N=5,
):
    """ """
    lines = np.array(list(votes.keys()))
    counts = np.array(list(votes.values()))
    sort_indices = np.argsort(counts)
    return np.sort(lines[sort_indices][-keep_top_N:])


def is_monotonically_increasing(xs, coeffs):
    p_temp = np.poly1d(coeffs)
    derivative = p_temp.deriv()
    return np.all(derivative(xs) > 0)
