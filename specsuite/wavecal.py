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


# for im_idx in range(10):
#     measured_lines = np.random.uniform(0, 1, N_lines)

#     model_lines = np.append(
#         measured_lines, np.random.uniform(0, 1, N_fake)
#     )

#     model_lines = np.sort(model_lines)
#     measured_lines = np.sort(measured_lines)

#     measured_lines = p(measured_lines)

#     rounding = 3

#     hash_table = build_hash_table(model_lines, rounding=rounding)
#     votes = cast_votes(hash_table, measured_lines, rounding=rounding)
#     features = get_most_likely_features(
#         votes,
#         keep_top_N=5,
#     )

#     from itertools import combinations

#     min_residual = np.inf
#     best_pix = None

#     for comb in combinations(measured_lines, len(features)):

#         coeffs = []

#         baseline = np.polyfit(features, comb, order)
#         residual = 0

#         for idx in range(len(features) - order - 1):
#             x = features[idx : 3 + idx]
#             y = comb[idx : 3 + idx]
#             residual += np.sum(np.abs(np.polyfit(x, y, order) - baseline))

#         if residual < min_residual:
#             min_residual = residual
#             best_pix = comb

#     coeffs = np.polyfit(features, best_pix, order)
#     p_best = np.poly1d(coeffs)

#     fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True)

#     for x in measured_lines:
#         axs[0].axvline(x)
#     axs[0].set_yticks([])

#     for x in p_best(features):
#         axs[1].axvline(x)
#     axs[1].set_yticks([])

#     for x in p_best(model_lines):
#         axs[2].axvline(x)
#     axs[2].set_yticks([])

#     axs[2].set_xlim(0, 1)
#     axs[2].set_xlim(np.min(measured_lines), np.max(measured_lines))
#     axs[2].set_xlabel("Line Location [pseudo-pixels]")

#     xmin, xmax = axs[2].get_xlim()

#     axs[0].text(xmin + (xmax-xmin)*0.01, 0.8, "Model Lines", fontsize=17, fontweight="bold")
#     axs[1].text(xmin + (xmax-xmin)*0.01, 0.8, "Most Likely Features", fontsize=17, fontweight="bold")
#     axs[2].text(xmin + (xmax-xmin)*0.01, 0.8, "Observed Lines (Transformed)", fontsize=17, fontweight="bold")

#     axs[0].set_title(f"Applied Transform: {np.round(original_coeffs, 2)}\nFitted Transform: {np.round(coeffs, 2)}")

#     plt.savefig(f"frames/{im_idx}.png", bbox_inches="tight")
#     plt.clf()
#     plt.close()

#     print(np.round(coeffs, 4))
