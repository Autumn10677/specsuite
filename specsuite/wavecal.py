import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from math import sqrt
from scipy.special import erf
from tqdm import tqdm
from astropy.stats import mad_std


def compute_triplet_values_from_indices(
    lines_round: np.ndarray,
    triplet_idx: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Using a list of rounded line positions, this function calculates
    the relative distances of each possible triplet of points. It
    also returns some related, useful arrays that can be used for
    filtering. All calculations are vectorized to reduce runtime.

    Parameters:
    -----------
    lines_round :: np.ndarray
        A 1D array containing rounded line positions.
    triplet_idx :: np.ndarray
        A 2D array (N x 3) containing the indices of each line that
        make up the triplet. The length N should be equal to the total
        number of valid triplets that can be formed from 'lines_round',
        and each entry should be a value representing an index location
        in the 'lines_round' array.

    Returns:
    --------
    b1 :: np.ndarray
        A 1D array containing all of the first base points.
    b2 :: np.ndarray
        A 1D array containing all of the second base points.
    p :: np.ndarray
        A 1D array containing all of the reference points.
    d :: np.ndarray
        A 1D array containing all of the scale-normalized distances
        calculated using b1, b2, and p. If the calculated scale is
        0.0 or the reference point is the same as one of the base
        points, the corresponding entry is replaced with a NaN value.
    valid :: np.ndarray
        A 1D array filled with boolean entries indicating whether the
        calculated distance is a NaN or not.
    i :: np.ndarray
        The indices of all b1 values in the 'triplet_idx' array.
    j :: np.ndarray
        The indices of all b2 values in the 'triplet_idx' array.
    k :: np.ndarray
        The indices of all p values in the 'triplet_idx' array.
    """

    # Allow for vectorized operations
    i = triplet_idx[:, 0]
    j = triplet_idx[:, 1]
    k = triplet_idx[:, 2]
    b1 = lines_round[i]
    b2 = lines_round[j]
    p = lines_round[k]

    # Calculates the scales and filters for invalid combinations
    denom = b2 - b1
    valid = ~np.isclose(denom, 0.0) & (p != b1) & (p != b2)

    # Calculates distances, invalid distances are given a NaN
    d = np.empty_like(denom, dtype=float)
    d[:] = np.nan
    d[valid] = (p[valid] - b1[valid]) / denom[valid]

    return b1, b2, p, d, valid, i, j, k


def sigma_d_from_triplets(
    b1: np.ndarray,
    b2: np.ndarray,
    p: np.ndarray,
    sigma_measured: float,
) -> np.ndarray:
    """
    Calculates the uncertainty in relative distance
    measurements for a given triplet of points.

    Parameters:
    -----------
    b1 :: np.ndarray
        A 1D array containing all of the first base points.
    b2 :: np.ndarray
        A 1D array containing all of the second base points.
    p :: np.ndarray
        A 1D array containing all of the reference points.
    sigma_measured :: float
        An estimate of the uncertainty in line positions.
        Assumes that the same error applies to all points.

    Returns:
    --------
    sigma_d :: np.ndarray
        A 1D array representing the uncertainty in distance
        measurements for each triplet.
    """

    # Calculates chunks of approximate error equation
    denom = b2 - b1
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = 1.0 / (denom**2)
        term2 = ((b2 - p) ** 2 + (p - b1) ** 2) / (denom**4)

    # Handles invalid errors by setting them to infinity
    var = sigma_measured**2 * (term1 + term2)
    sigma_d = np.sqrt(var)
    sigma_d[~np.isfinite(sigma_d)] = np.inf

    return sigma_d


def calculate_evidence(
    valid_keys: np.ndarray,
    d_obs: np.ndarray,
    sig_d: np.ndarray,
    pad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Bayesian evidence (i.e., the probability
    of a given key being calculated integrated over all
    triplet pairs).

    Parameters:
    -----------
    valid_keys :: np.ndarray
        All possibe keys (rounded distances) across all triplet
        pairs.
    d_obs :: np.ndarray
        The calculated relative distances corresponding to each
        triplet pair.
    sig_d :: np.ndarray
        The calculated error in each distance measurement provided
        in 'd_obs'.
    pad :: float
        The size of half a bin in the geometric hashing routine.
        This should be equivalent to 0.5 * 10^(-rounding).

    Returns:
    --------
    evidence :: np.ndarray
        A 1D array containing the probability that each key is measured
        across all possible triplet combinations.
    prob_mass :: np.ndarray
        A 2D array containing the probability mass for all possible
        triplet combinations.
    """

    # Shape broadcasting to make operations easier later on
    DOBS = d_obs[:, None]
    SIG = sig_d[:, None]
    KEYS = valid_keys[None, :]

    # This determines the 'size' of bin to integrate over
    L = KEYS - pad
    R = KEYS + pad

    # Approximates indefinite integral over normal distribution
    denom = SIG * sqrt(2)
    z1 = (R - DOBS) / denom
    z0 = (L - DOBS) / denom
    prob_mass = 0.5 * (erf(z1) - erf(z0))

    # Sums probability mass across triplets to get evidence mass per key
    evidence_raw = prob_mass.sum(axis=0)
    total = evidence_raw.sum()
    if total == 0:
        evidence = np.ones_like(evidence_raw) / float(len(evidence_raw))
    else:
        evidence = evidence_raw / total

    return evidence, prob_mass


def cast_votes(
    n_model_points: int,
    n_data_points: int,
    prob_mass: np.ndarray,
    evidence: np.ndarray,
    valid_keys: np.ndarray,
    hash_table: dict,
    di_v: np.ndarray,
    dj_v: np.ndarray,
    dk_v: np.ndarray,
):
    """
    Casts votes for each possible combination of points
    using a traditional geometric hashing scheme. If a
    pair of triplets produces a key found in the hash
    table, then the data points will be added as a possible
    match for each of model points listed for that entry
    in the hash table. However, the votes are weighted by
    the Bayesian probability of that match being correct.

    Parameters:
    -----------
    n_model_points :: int
        The total number of model points.
    n_data_points :: int
        The total number of data points.
    prob_mass :: np.ndarray
        A 2D array containing the probability mass for all possible
        triplet combinations.
    evidence :: np.ndarray
        A 1D array containing the probability that each key is measured
        across all possible triplet combinations.
    valid_keys :: np.ndarray
        A list of all possible keys in the hash table. This is stored
        in a separate list to prevent unecessary duplicate lookups.
    hash_table :: dict
        A hash table where each key corresponds to the relative distance
        calculated from three points.
    di_v :: np.ndarray
        The indices of all b1 values in the data triplet array.
    dj_v :: np.ndarray
        The indices of all b2 values in the data triplet array.
    dk_v :: np.ndarray
        The indices of all p values in the data triplet array.

    Returns:
    --------
    votes :: np.ndarray
        The probability-weighted votes indicating which pairs of model and
        data points is most likely.
    """

    # Initialized with known size to prevent slow appending
    votes = np.zeros((n_model_points, n_data_points), dtype=float)
    weights_matrix = prob_mass * evidence[None, :]

    # Iterates over all valid keys (instead of triplet pairs)
    for k_index, key in enumerate(valid_keys):

        # Only continues if key is 'valid' and a valid triplet exists
        if key not in hash_table:
            continue
        model_triplet_indices = hash_table[key]
        if model_triplet_indices.size == 0:
            continue

        # Only continues if key has non-zero probability mass
        weights_for_key = weights_matrix[:, k_index]
        if np.all(weights_for_key <= 0):
            continue

        # For each model triplet, add weights_for_key to appropriate votes rows
        for m1_idx, m2_idx, m3_idx in model_triplet_indices:
            np.add.at(votes[m1_idx], di_v, weights_for_key)
            np.add.at(votes[m2_idx], dj_v, weights_for_key)
            np.add.at(votes[m3_idx], dk_v, weights_for_key)

    return votes


def match_features(
    raw_data_lines: np.ndarray,
    raw_model_lines: np.ndarray,
    iterations: int = 10,
    rounding: int = 3,
    sigma: float = 2.0,
    order: int = 2,
    debug: bool = False,
):
    """
    Attempts to find the most probable matches in lines
    bewteen two lists. This function assumes that the
    model lines have no (or negligible) error, and that
    all data lines have a constant error described by
    'sigma'.

    Parameters:
    -----------
    raw_data_lines :: np.ndarray
        A 1D list of lines indicating the pixel positions
        of line emissions.
    raw_model_lines :: np.ndarray
        A 1D list of line indicating the known wavelengths
        of some of the observed data lines. This list can
        be shorter or longer than 'raw_data_lines,' but
        this will impact the accuracy of the voting routine.
    iterations :: int
        The number of iterations to repeat feature matching.
        For well-behaved examples, the votes will converge
        to a consistent answer after a handful of iterations.
        However, excessive looping can lead to a gradual break
        in the accuracy of the calculated matches.
    rounding :: int
        How many decimal places to measure relative distances
        to. For example, using 'rounding=3' will calculate
        distances to three decimal places.
    sigma :: float
        The uncertainty (in the same units as 'raw_data_lines')
        of data line locations.
    order :: int
        The polynomial order to fit to the matched features. We
        highly recommend keeping this at 'order=2'.
    debug :: bool
        Allows for diagnostic plots to be shown.

    Returns:
    --------
    votes :: np.ndarray
        A 2D array showing the probability of a given data line
        being paired with a given model line.
    """

    # Rounds line lists for use in a geometric hashing algorithm
    data_lines_round = np.round(raw_data_lines, rounding)
    model_lines_round = np.round(raw_model_lines, rounding)

    # Pad used for integration over bins (half of bin size)
    pad = 0.5 * 10 ** (-rounding)

    n_model_points = len(model_lines_round)
    n_data_points = len(data_lines_round)
    model_triplet_idx = np.array(
        list(combinations(range(n_model_points), 3)), dtype=int
    )
    data_triplet_idx = np.array(list(combinations(range(n_data_points), 3)), dtype=int)

    # This tries to iteratively improve geometric hashing with few protetctions
    # Future work should probably attempt to perform some sanity checks
    for iteration in tqdm(
        range(iterations),
        desc="Matching Features",
    ):

        # Calculates relevant triplet values for model points
        bm1, bm2, bp, d_model, valid_model_mask, mi, mj, mk = (
            compute_triplet_values_from_indices(
                lines_round=model_lines_round,
                triplet_idx=model_triplet_idx,
            )
        )

        # Filters out model triplets that did not produce a valid key
        valid_model_indices = np.where(valid_model_mask)[0]
        if len(valid_model_indices) == 0:
            raise RuntimeError(
                "No valid model triplets; check rounding or model points."
            )
        d_model_valid = np.round(d_model[valid_model_indices], rounding)

        # store model triplets as index triples to avoid repeated value lookups
        hash_table = {}
        unique_keys, inv = np.unique(d_model_valid, return_inverse=True)
        for k_idx, key in enumerate(unique_keys):
            locs = valid_model_indices[np.where(inv == k_idx)[0]]
            # model triplet index triples (indices into model_lines arrays)
            hash_table[key] = model_triplet_idx[locs]  # shape (M,3)

        valid_keys = np.array(sorted(hash_table.keys()))
        if valid_keys.size == 0:
            raise RuntimeError(
                "No keys in model hash; try larger model or different rounding."
            )

        # Calculates relevant triplet values for model points
        db1, db2, dp, d_data, valid_data_mask, di, dj, dk = (
            compute_triplet_values_from_indices(
                lines_round=data_lines_round,
                triplet_idx=data_triplet_idx,
            )
        )

        # Filters out data triplets that did not produce a valid key
        valid_data_indices = np.where(valid_data_mask)[0]
        if len(valid_data_indices) == 0:
            raise RuntimeError("No valid data triplets; check rounding or data points.")

        # Trims arrays to only the valid subset of triplets
        db1_v = db1[valid_data_indices]
        db2_v = db2[valid_data_indices]
        dp_v = dp[valid_data_indices]
        d_obs = d_data[valid_data_indices]
        di_v = di[valid_data_indices]
        dj_v = dj[valid_data_indices]
        dk_v = dk[valid_data_indices]

        # Calculates error in relative distances
        sig_d = sigma_d_from_triplets(
            b1=db1_v,
            b2=db2_v,
            p=dp_v,
            sigma_measured=sigma,
        )

        # Filters out triplets with errors of 0.0 or np.inf
        finite_mask = (sig_d > 0) & np.isfinite(sig_d)
        if not np.all(finite_mask):
            db1_v = db1_v[finite_mask]
            db2_v = db2_v[finite_mask]
            dp_v = dp_v[finite_mask]
            d_obs = d_obs[finite_mask]
            sig_d = sig_d[finite_mask]
            di_v = di_v[finite_mask]
            dj_v = dj_v[finite_mask]
            dk_v = dk_v[finite_mask]
        n_data_triplets = len(d_obs)
        if n_data_triplets == 0:
            raise RuntimeError("No usable data triplets after sigma filtering.")

        # Calculates probability of each key being found given all triplets
        evidence, prob_mass = calculate_evidence(
            valid_keys=valid_keys,
            d_obs=d_obs,
            sig_d=sig_d,
            pad=pad,
        )

        # Casts votes to match model lines to data lines
        votes = cast_votes(
            n_model_points=n_model_points,
            n_data_points=n_data_points,
            prob_mass=prob_mass,
            evidence=evidence,
            valid_keys=valid_keys,
            hash_table=hash_table,
            di_v=di_v,
            dj_v=dj_v,
            dk_v=dk_v,
        )

        # Extracts data / model lines data with corresponding match scores
        best_data_idx = np.argmax(votes, axis=1)
        xs = raw_model_lines.copy()
        ys = raw_data_lines[best_data_idx]

        xs_fit = np.array(xs)
        ys_fit = np.array(ys)

        # (FIXME: For now, holding these constant for development)
        max_removals = int(len(xs_fit) / 2)
        removed = 0
        thresh = 5.0

        # Iteratively removes outliers before fitting polynomial
        for _ in range(max_removals):
            if len(xs_fit) < 3:
                break

            # (FIXME: probably a better way to do this than a linear fit?)
            p_extracted_linear = np.poly1d(np.polyfit(xs_fit, ys_fit, deg=1))
            residuals = np.abs(p_extracted_linear(xs_fit) - ys_fit)

            # In case of zero-like residuals, all scales are 1.0
            scl = mad_std(residuals)
            if scl == 0:
                scl = np.std(residuals) if np.std(residuals) > 0 else 1.0

            # Finds which point is the largest outlier
            residuals_scaled = residuals / scl
            max_idx = np.argmax(residuals_scaled)

            # Removes points exceeding the threshold
            if residuals_scaled[max_idx] > thresh:
                xs_fit = np.delete(xs_fit, max_idx)
                ys_fit = np.delete(ys_fit, max_idx)
                removed += 1
            else:
                break

        # Fits a polynomial to the remaining points
        if len(xs_fit) >= order + 1:
            p_extracted = np.poly1d(np.polyfit(xs_fit, ys_fit, deg=order))
        else:
            p_extracted = np.poly1d(
                np.polyfit(xs_fit, ys_fit, deg=min(1, max(1, len(xs_fit) - 1)))
            )

        votes = (votes - np.min(votes)) / (np.max(votes) - np.min(votes))

        # Optional plotting
        if debug:
            fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
            for i in range(n_model_points):
                for j in range(n_data_points):
                    axs[2].scatter(
                        raw_model_lines[i],
                        raw_data_lines[j],
                        color="k",
                        alpha=votes[i][j],
                    )
            for x in raw_model_lines:
                axs[0].axvline(x, color="k")
            for x in raw_data_lines:
                axs[1].axvline(x, color="k")
            axs[0].set_title(f"Iteration: {iteration}")
            axs[0].set_ylabel("Model Lines")
            axs[1].set_ylabel("Data Lines")
            axs[2].plot(xs, p_extracted(xs), color="red", ls="--")
            axs[2].set_ylabel("Inferred Fit")
            plt.tight_layout()
            plt.show()

            # Modify 'model lines' using extracted polynomial transformation
            raw_model_lines = p_extracted(raw_model_lines)
            model_lines_round = np.round(
                raw_model_lines, rounding, out=np.empty_like(model_lines_round)
            )

    return votes
