import math
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import scipy.signal as signal
from astropy.stats import mad_std
from joblib import Parallel, delayed
import warnings

from .utils import plot_image

import sys

sys.tracebacklimit = 0


def find_cal_lines(
    image: np.ndarray,
    std_variation: float = 50.0,
    row: int = None,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds pixel positions of spectral lines in a provided image. The
    image does not need to range from any specific wavelength to
    another. The only requirement is that the lines are located
    vertically on the image plane.

    Parameters:
    -----------
    image :: np.ndarray
        Image with calibration lines.
    std_variation :: float
        How many standard deviations a peak must exceed the baseline to
        be counted as a line.
    row :: int
        Which row to pull a 1D profile from. If no argument is
        provided, this function will default to the row at the center
        of the cross-dispersion axis.
    debug :: bool
        Allows for a diagnostic plot to be shown.

    Returns:
    --------
    non_zero_indices :: np.ndarray
        List of pixel locations for detected lines.
    rel_intensity :: np.ndarray
        List of the relative intensity of the detected lines.
    """

    if row is None:
        row = len(image) // 2

    # Extracts a single strip from the image
    sample_light = image[row, :]
    sample_locs = np.arange(len(sample_light))

    # Masks out NaN values
    sample_locs = sample_locs[~np.isnan(sample_light)]
    sample_light = sample_light[~np.isnan(sample_light)]

    # Filters light to only include peaks above a certain level
    sample_std = mad_std(sample_light)
    filtered_light = np.array(
        [i if i > std_variation * sample_std else 0 for i in sample_light]
    ).astype(float)

    if np.max(filtered_light) == 0.0:
        raise ZeroDivisionError(
            f"No pixels were found above the provided threshold ({std_variation})"
        )

    filtered_light /= np.max(filtered_light)

    non_zero_indices = []
    rel_intensity = []
    i = 0

    # Iterates over each pixel in 1D spectra
    while i < len(filtered_light):

        # Finds non-zero values
        if filtered_light[i] != 0:

            # Start of spectral line
            start = i

            # Runs while non-zero
            while i < len(filtered_light) and filtered_light[i] != 0:
                i += 1

            # End of spectral line
            end = i - 1

            if sample_locs[start] != sample_locs[end]:
                avg_index = (sample_locs[start] + sample_locs[end]) / 2
            else:
                avg_index = sample_locs[start]

            non_zero_indices.append(avg_index)
            rel_intensity.append(filtered_light[i - 1])

        else:
            i += 1

    if debug:

        # Create stacked plots
        plt.rcParams["figure.figsize"] = (10, 6)

        # Upper plot: Filtered light with detected peaks
        plt.subplot(2, 1, 1)
        for peak in non_zero_indices:
            plt.axvline(peak, color="k", ls="--", alpha=0.5)
        plt.plot(sample_locs, filtered_light)
        plt.xlim(0, len(image[0]))
        plt.ylim(0, 1.1)
        plt.xlabel("Pixel")
        plt.ylabel("Normalized and Filtered Flux")
        plt.title(rf"Found {len(non_zero_indices)} Lines")

        # Lower plot: Original image strip
        plt.subplot(2, 1, 2)
        plt.imshow(image, cmap="gray", aspect="auto", norm="log", origin="lower")
        plt.axhline(row, color="red", ls="--")

        plt.tight_layout()
        plt.show()

    return np.array(non_zero_indices), np.array(rel_intensity)


def combine_within_tolerance(values: np.ndarray, tolerance: float) -> np.ndarray:
    """
    Takes a user-given list and combines values that are within a given
    tolerance. This is helpful for when a dense sample of features is
    non-ideal and should be combined into a single feature.

    Parameters:
    -----------
    values :: np.ndarray
        List of values to be analyzed.
    tolerance :: float
        The theshold at which two data points should be combined.

    Returns:
    --------
    combined_values :: list
        List of values where close points have been averaged and
        combined.
    """

    # Silently handled here in case users provide negative tolerance
    try:
        tolerance = np.abs(tolerance)
        values = sorted(list(values))
        combined_values = []
        temp_group = [values[0]]
    except IndexError:
        return np.array([])
    except TypeError:
        return np.array([])

    try:
        # Iterates over each value
        for i in range(1, len(values)):

            # Checks whether values are within a tolerance
            if np.abs(values[i] - temp_group[-1]) <= tolerance:
                temp_group.append(values[i])

            # Combines close points and resents temporary list
            else:
                combined_values.append(np.mean(temp_group))
                temp_group = [values[i]]

        # Add the last group
        combined_values.append(np.mean(temp_group))

    except TypeError:
        return values

    return np.array(combined_values).astype(float)


def generate_warp_model(
    image: np.ndarray,
    guess: np.ndarray,
    tolerance: int = 16,
    line_order: int = 2,
    warp_order: int = 1,
    ref_idx: int = None,
    debug: bool = False,
) -> list:
    """
    Models how straight vertical lines in a wavelength calibration
    image are being warped. Assumes a relatively low amount of
    straight-line warping and that these lines are continuous. This
    function allows for the type of warping to change along the
    horizontal axis of the detector. This is functionally achieved by
    binning down a given image and identifying the brightest pixels
    across the vertical axis of a detector in a small region around
    each line.

    Parameters:
    -----------
    image :: np.ndarray
        A 2D arclamp image with strong emission features.
    guess :: np.ndarray
        The pixel positions (along the dispersion axis) of strong
        emissions features in the provided image. These can be slightly
        wrong, as the 'tolerance' determines how broad of a region to
        analyze around each of these guesses.
    tolerance :: int
        Number of pixels around your guess to use for building a warp
        model. The tolerance should at least be large enough that the
        entire emission line falls within a given window, but it is
        often better to provide a slightly larger tolerance if you
        are uncertain in the accuracy of your guesses.
    line_order :: int
        Order of x-warping (i.e. How does our vertial warping change
        with x-position along the detector).
    warp_order :: int
        Order of y-warping (i.e. What shape does a straight line
        projected onto our detector take on).
    ref_idx :: int
        Which row (along cross-dispersion axis) to use for measuring
        the relative changes from row to row. This choice should be
        near the brightest portion of the emission line.
    debug :: bool
        Allows for diagnostic plots to be displayed.

    Returns:
    --------
    warp_models :: list
        Collection of models describing how y-warping coefficients
        change as a function of x.
    """

    if len(guess) == 0:
        warnings.warn(
            "The provided 'guess' array is empty and a warp model could not be generated"  # noqa: E501
        )
        return None

    if not isinstance(tolerance, int):
        warnings.warn(
            f"'tolerance' must be an int (not {type(tolerance)}), using default value\n"  # noqa: E501
        )
        tolerance = 16

    if not isinstance(line_order, int):
        warnings.warn(
            f"'line_order' must be an int (not {type(warp_order)}), using default value\n"  # noqa: E501
        )
        line_order = 2

    if not isinstance(warp_order, int):
        warnings.warn(
            f"'warp_order' must be an int (not {type(warp_order)}), using default value\n"  # noqa: E501
        )
        warp_order = 1

    coeff_list = np.array([])
    if ref_idx is None:
        ref_idx = len(image) // 2

    for loc in guess:

        subim = image[:, int(loc) - tolerance : int(loc) + tolerance]
        ref_row = subim[ref_idx, :]

        lag_rows = np.zeros(len(subim))
        lag_list = np.zeros(len(subim))

        for idx, row in enumerate(subim):
            correlation = signal.correlate(
                ref_row - np.mean(ref_row), row - np.mean(row), mode="full"
            )
            lags = signal.correlation_lags(len(ref_row), len(row), mode="full")
            sort_idx = np.argsort(correlation)
            lag = -lags[sort_idx][-1]
            lag_list[idx] = lag
            lag_rows[idx] = idx

        lag_list += loc

        coeffs = np.polyfit(lag_rows, lag_list, line_order)
        coeff_list = np.concatenate([coeff_list, coeffs], axis=0)

    coeff_list = coeff_list.reshape((len(guess), line_order + 1)).T

    warp_coeffs = []
    warp_models = []
    for idx in range(len(coeff_list)):
        warp_coeffs.append(np.polyfit(guess, coeff_list[idx], warp_order))
        warp_models.append(np.poly1d(warp_coeffs[-1]))

    # Optional plotting
    if debug:

        # Calculates the approximate number of subplots
        max_cols = 3
        n_plots = len(guess)
        n_rows = math.ceil(n_plots / max_cols)

        # Initializes figure
        fig, axes = plt.subplots(n_rows, max_cols, figsize=(max_cols * 4, n_rows * 2))
        axes = axes.flatten()

        cd_pix = range(len(image))
        grid_y_locs = np.linspace(min(cd_pix), max(cd_pix), 5)

        for i, loc in enumerate(guess):

            ax = axes[i]

            grid_x_locs = np.linspace(loc - tolerance, loc + tolerance, 10)

            for x in grid_x_locs:
                line_model = np.poly1d([p(x) for p in warp_models])
                ax.plot(line_model(cd_pix), cd_pix, color="white")

            for y in grid_y_locs:
                ax.axhline(y, color="white")

            ax.imshow(
                image,
                aspect="auto",
                norm="log",
                cmap="inferno",
                origin="lower",
                interpolation="none",
            )
            ax.set_xlabel("Dispersion (pix)")
            ax.set_ylabel("Cross-Dispersion (pix)")
            ax.set_xlim(loc - tolerance, loc + tolerance)
            ax.set_ylim(0, len(image))

        # Hide any unused axes
        for j in range(n_plots, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        plt.show()

    return warp_models


def dewarp_image(
    image: np.ndarray, models: list, debug: bool = False, progress: bool = False
) -> np.ndarray:
    """
    Uses a 'warp model' to undo distortion in a 2D image. This is done
    by rebinning each row onto a consistent wavelength bin proportional
    to the overlapping area between the raw image data and a warped set
    of new pixel locations.

    Parameters:
    -----------
    image :: np.ndarray
        Image that is horizontally warped according
        to a set of models the user also provides.
    models :: list
        List of models that describe how vertical lines
        are warped along the image's horizontal axis.
    debug :: bool
        Enables diagnostic plots.
    progress :: bool
        Enables a progress bar.

    Returns:
    --------
    unwarped_image :: np.ndarray
        A dewarped version of the provided image.
    """

    image = np.array(image)

    if len(image.shape) != 2:
        warnings.warn(
            f"The provided image must be a single 2D array (not a {len(image.shape)}D array)"  # noqa: E501
        )
        return image

    # Defines an array of pixel edges for each raw data pixel
    edges = np.array(range(len(image[0]) + 1)) - 0.5

    # Calculates the edges of each warped pixel across the detector
    rows = np.array(range(len(image)))
    w_edge_map = np.array(
        [
            models[0](edges) * r**2 + models[1](edges) * r + models[2](edges)
            for r in rows
        ]
    )

    # Extracts arrays of data pixel edges
    pix_l = edges[:-1]
    pix_r = edges[1:]

    # Function for flux remapping
    def process_row(row):

        # Get the warped edges for this row
        box_l = w_edge_map[row][:-1]
        box_r = w_edge_map[row][1:]

        # Calculates area overlap between each warped and original data pixel
        overlaps_matrix = np.maximum(
            0, np.minimum(box_r[:, None], pix_r) - np.maximum(box_l[:, None], pix_l)
        )

        # Uses our overlap matrix and original flux to fill in the dewarped image
        fluxes = (overlaps_matrix * image[row]).sum(axis=1)

        return fluxes

    # Uses parallel processing to dewarp each row in tandem
    unwarped_image = np.array(
        Parallel(n_jobs=-1)(
            delayed(process_row)(row)
            for row in tqdm(
                rows, desc="dewarping", position=0, leave=True, disable=not progress
            )
        )
    )

    # Plotting
    if debug:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

        # First plot: Total row flux for both images
        ax1.plot(
            np.sum(image, axis=1), linewidth=5, color="black", label="Original Image"
        )
        ax1.plot(
            np.sum(unwarped_image, axis=1),
            linewidth=1,
            color="red",
            label="Unwarped Image",
        )
        ax1.set_ylabel("Total Row Flux", fontsize=13)
        ax1.set_yscale("log")
        ax1.legend()

        # Second plot: Absolute flux difference
        ax2.plot(
            np.abs(np.sum(image, axis=1) - np.sum(unwarped_image, axis=1)),
            color="black",
            label="Flux Difference",
        )
        ax2.set_xlabel("Row Pixel (y)", fontsize=13)
        ax2.set_ylabel("Absolute Flux Difference", fontsize=13)
        ax2.set_yscale("log")

        # Remove space between plots
        plt.subplots_adjust(hspace=0.2)
        plt.show()

    return unwarped_image


def generate_effpix_map(
    xs: np.ndarray, rows: np.ndarray, models: np.ndarray
) -> np.ndarray:
    """
    Generates a 2D map of the effective location of each pixel center
    across the detector. It uses models that describe the warping
    across the detector as a function of row and column positions.

    Parameters:
    -----------
    xs :: np.ndarray
        A 1D array holding all column pixel positions.
    rows :: np.ndarray
        A 1D array holding all row pixel positions.
    models :: np.ndarray
        An array holding how light warping changes as a function of
        column and row pixel positions. This function assumes that each
        fitting coefficient changes according to a linear relationship.

    Returns:
    --------
    effpix_map :: np.ndarray
        A 2D array detailing the effective location of each pixel center.
    """

    # Initializes array for map data
    effpix_map = []

    # Calculates each effective pixel in a given row
    for row in rows:
        simplified_model = models[0] * row**2 + models[1] * row + models[2]
        model_coeffs = simplified_model.coeffs
        effpix_map.append((xs - model_coeffs[-1]) / model_coeffs[0])

    return np.array(effpix_map)


def extract_background(
    images: np.ndarray,
    warp_model: list,
    mask_region: tuple = (None, None),
    return_spectrum: bool = False,
    progress: bool = False,
    debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the sky background for a series of exposures where a trace
    should be masked out. It uses a 'warp model' that describes how
    straight emission lines are 'bent' on the imaging plane. This
    function assumes that flatfielding has been applied and that sky
    emissions are approximately uniform along the spatial axis.

    Parameters:
    -----------
    images :: np.ndarray
        A series of exposures to extract backgrounds from.
    warp_model :: list
        List of models that describe how vertical lines
        are warped along the image's horizontal axis.
    mask_region :: tuple
        The pixel locations (vertical) to mask out during the
        background extraction.
    return_spectrum :: bool
        Whether or not to return the super-sampled spectra used for
        interpolating the sky background. If 'True', three additional
        arguments are returned. These are the effective pixel
        locations, effective flux, and the 'effpix map'.
    progress :: bool
        Controls whether or not progress bars are shown.
    debug :: bool
        Allows for diagnostic plots to be shown.

    Returns:
    --------
    background_images :: np.ndarray
        A series of images representing the approximated sky background
        for each image.
    supersampled_effpix :: np.ndarray
        The grid of 'effective pixel locations' that the sky emission
        spectra was extracted for. This is essentially a flattened
        version of the 'effpix_map'.
    supersampled_spectra :: np.ndarray
        The effective flux found for each of the effective pixel
        locations.
    effpix_map :: np.ndarray
        A 2D array with the same size as an individual image. The value
        at each pixel represents the 'effective pixel location' for
        that given pixel.
    """

    # Ensure arrays are 3D and valid
    science_images = images.copy()
    if len(science_images.shape) == 2:
        science_images = np.array([science_images])
    if None in mask_region:
        mask_region = (-1, -1)

    # Generate geometry info
    n_rows, n_cols = science_images.shape[1:]
    xs = np.arange(n_cols)
    xs_edges = np.arange(n_cols + 1) - 0.5
    rows = np.arange(n_rows)

    # Compute effective pixel positions
    effpix_map = generate_effpix_map(xs, rows, warp_model)
    effpix_edge_map = generate_effpix_map(xs_edges, rows, warp_model)

    # Apply trace mask
    signal_mask = np.ones_like(science_images)
    signal_mask[:, mask_region[0] : mask_region[1]] = np.nan
    effpix_map *= signal_mask[0]
    science_images *= signal_mask

    # Flatten to 1D spectra and filter
    shape = science_images.shape
    supersampled_effpix = effpix_map.flatten()
    supersampled_spectra = science_images.reshape(
        (
            shape[0],
            -1,
        )
    )

    valid_mask = ~np.isnan(supersampled_effpix)
    effpix_sorted = np.sort(supersampled_effpix[valid_mask])
    sorted_idx = np.argsort(supersampled_effpix[valid_mask])
    spectra_sorted = supersampled_spectra[:, valid_mask][:, sorted_idx]

    # Initializes array for backgrounds
    background_images = np.zeros(science_images.shape)
    for idx, image in enumerate(
        tqdm(science_images, desc="Extracting Background", disable=(not progress))
    ):

        # Pulls the background flux data for any non-masked region
        background_flux = spectra_sorted[idx]

        # Function for flux remapping
        def process_row(row):
            background_row = np.zeros(len(image[0]))
            indices = np.searchsorted(effpix_sorted, effpix_edge_map[row])
            for idx in range(len(indices) - 1):
                if len(background_flux[indices[idx] : indices[idx + 1] + 1]) == 0:
                    background_row[idx] = np.nan
                else:
                    background_row[idx] = np.median(
                        background_flux[indices[idx] : indices[idx + 1] + 1]
                    )
            return background_row

        # Uses parallel processing to extract background emissions for rows in tandem
        background = np.array(
            Parallel(n_jobs=-1)(delayed(process_row)(row) for row in rows)
        )
        background_images[idx] = background

    if debug:

        VMIN, VMAX = np.min(images[0]), np.max(images[0])

        plot_image(
            images[0],
            title="Original Exposure",
            norm="log",
            vmin=VMIN,
            vmax=VMAX,
        )
        plot_image(
            background_images[0],
            title="Extracted Background",
            norm="log",
            vmin=VMIN,
            vmax=VMAX,
        )
        plot_image(
            images[0] - background_images[0],
            title="Corrected Exposure",
            norm="log",
            vmin=VMIN,
            vmax=VMAX,
        )

    if return_spectrum:
        return background_images, supersampled_effpix, supersampled_spectra, effpix_map
    return background_images
