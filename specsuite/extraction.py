import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import curve_fit

from .utils import _gaussian, _moffat, rebin_image_columns


def generate_spatial_profile(
    image: np.ndarray,
    profile: str = "moffat",
    profile_order: int = 7,
    bin_size: int = 8,
    repeat: bool = True,
    debug: bool = False,
):
    """
    Generates a 'spatial profile' as outlined in Horne (1986).
    Spatial profiles predict the likelihood that a photon would
    land at a given cross-dispersion location for each wavelength.
    This function assumes that the dispersion axis is located
    along the x-axis.

    Parameters:
    -----------
    image :: np.ndarray
        The image that a spatial profile is fit to.
    profile :: str
        Name of the type of profile to fit for. Currently, the
        only valid options are...
            - moffat
            - gaussian
    profile_order :: int
        The order of the polynomial used to fit to each constant
        in the specified spatial profile (i.e., along the dispersion
        axis, the mean evolve as what order of polynomial?)
    bin_size :: int
        Size of each bin used for 'binning down' the provided image
        before fitting.
    repeat :: bool
        Allows the initial fit to each parameter to influence the
        initial guesses in a second series of fits.
    debug :: bool
        Allows for optional debugging plots to be shown.
    """

    assert profile in ["moffat", "gaussian"], f"'{profile}' is not a valid profile..."

    # Stores fitting information (function, p0, bounds) for each model
    profile_dict = {
        "gaussian": [_gaussian, [0.5, -1, 2.5], [[0, 0, 0], [1, len(image), 10]]],
        "moffat": [
            _moffat,
            [0.5, -1, 5, 0.01],
            [[0, 0, 4, 0], [1, len(image), 20, np.inf]],
        ],
    }

    # Extracts profile information
    profile_function = profile_dict[profile][0]
    p0 = profile_dict[profile][1]
    bounds = profile_dict[profile][2]

    # Bins down image to mitigate cosmic rays
    binned_image = rebin_image_columns(image, bin_size)
    binned_image /= np.clip(np.sum(binned_image, axis=0), 1, None)

    # Creates arrays for binned and unbinned indexes
    rows = np.array(range(len(image))).astype(int)
    cols = np.array(range(len(image[0]))).astype(int)
    cols_binned = (bin_size * (np.array(range(len(binned_image[0]))) + 0.5)).astype(int)

    coeffs = []
    parameters = []
    successful_cols = []

    # Fits profile to each binned column
    for run_number in range(2 if repeat else 1):

        for idx in range(len(cols_binned)):

            try:
                y = binned_image[:, idx]

                if run_number == 0:
                    p0[1] = np.argmax(y)
                elif run_number == 1 and idx == 0:
                    p0 = np.median(parameters, axis=0)

                    parameters = []
                    successful_cols = []

                popt, _ = curve_fit(profile_function, rows, y, p0=p0, bounds=bounds)
                parameters.append(popt)
                successful_cols.append(cols_binned[idx])

            # Prevents printout if fit does not converge
            except RuntimeError:
                pass

    parameters = np.array(parameters).T

    # Fits for how PSF constants evolve along dispersion axis
    for idx in range(len(parameters)):
        p = np.poly1d(np.polyfit(successful_cols, parameters[idx], profile_order))
        coeffs.append(p(cols))

        if debug:
            plt.rcParams["figure.figsize"] = (12, 4)
            plt.scatter(successful_cols, parameters[idx])
            plt.plot(successful_cols, p(successful_cols))
            plt.show()

    coeffs = np.array(coeffs).T

    # Generates spatial profile
    P = np.zeros(image.shape)
    for idx in range(len(coeffs)):
        xs = np.array(range(len(P)))
        P[:, idx] = profile_function(xs, *coeffs[idx])
    P /= np.sum(P, axis=0)

    return P


def boxcar_extraction(
    images: np.ndarray,
    backgrounds: np.ndarray,
    RN: float | np.ndarray = 0.0,
    debug: bool = False,
):
    """
    Performs a simple boxcar extraction on an image
    (or series of images). This assumes that both arrays
    of images of dimensions corresponding to...

        (cross-dispersion, dispersion)

    If that is not the case, please rotate your data arrays
    before feeding them into this function.

    Parameters:
    -----------
    images :: np.ndarray
        A 2D (or array of several 2D) science exposures that
        have been background subtracted.
    backgrounds :: np.ndarray
        A 2D (or array of several 2D) background exposures
        that have been subtracted off of your science images.
    RN :: float | np.ndarray
        The read noise associated with your detector.
    debug :: bool
        Allows for optional plotting.

    Returns:
    --------
    flux_array :: np.ndarray
        A 2D array containing the flux of each provided exposure.
        Has a shape of (image index, pixel position).
    error_array :: np.ndarray
        A 2D array containing the undertainty of each provided
        exposure. Has a shape of (image index, pixel position).
    """

    # Handles single-image exposures by wrapping them in a list
    if len(images.shape) != 3:
        images = np.array([images])
    if len(backgrounds.shape) != 3:
        backgrounds = np.array([backgrounds])

    # Checks that arrays are either 3D or a wrapped 2D exposure
    try:
        assert (len(images.shape) == 3) and (len(backgrounds.shape) == 3)
    except AssertionError:
        raise AssertionError("Both image arrays should be 2D or 3D.")

    # Assumes that 'images' and 'backgrounds' are 3D arrays
    flux_array = np.sum(images, axis=1)
    error_array = np.sqrt(np.sum(images + backgrounds + RN**2, axis=1))

    if debug:
        pixel_positions = np.array(range(len(flux_array[0])))

        plt.rcParams["figure.figsize"] = (12, 4)
        plt.errorbar(
            pixel_positions,
            flux_array[0],
            yerr=error_array[0],
            color="k",
            label="First Exposure",
            fmt="none",
        )
        plt.plot(
            pixel_positions,
            np.median(flux_array, axis=0),
            color="salmon",
            label="Median Exposure",
            zorder=-999,
        )
        plt.xlim(np.min(pixel_positions), np.max(pixel_positions))
        plt.xlabel("Pixel Position (Dispersion Axis)")
        plt.ylabel("Extracted Flux / Pixel")
        plt.legend()
        plt.show()

    return flux_array, error_array


def horne_extraction(
    images: np.ndarray,
    backgrounds: np.ndarray,
    profile: str = "moffat",
    profile_order: int = 3,
    RN: float | np.ndarray = 0.0,
    bin_size: int = 16,
    max_iter: int = 5,
    repeat: bool = True,
    debug: bool = False,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs a profile-weighted (Horne) extraction for a series
    of science exposures.

    Parameters:
    -----------
    images :: np.ndarray
        A single (or multiple) 2D exposures containing a point-source
        trace to extract flux from.
    backgrounds :: np.ndarray
        A single (or multiple) 2D exposures contianing the background
        subtracted off of the science exposures. Used for calculating
        the uncertainty of the reduction.
    profile :: str
        Which type of 1D profile to use for generating a spatial
        profile. Valid options are 'moffat' or 'gaussian'.
    profile_order :: int
        The polynomial order that describes how the 1D profile (in the
        fitted spatial profile) changes with pixel position along the
        dispersion axis.
    RN :: float | np.ndarray
        The read noise of your exposure. If the provided argument is a
        float, then every pixel will be assigned an equal read noise.
        Otherwise, if provided a 2D array, then each exposure will be
        assigned the corresponding value for that pixel.
    bin_size :: int
        The number of pixels (dispersion axis) to lump into a single
        bin when generating a spatial profile. Generally, a higher value
        increases the probability that 'generate_spatial_profile()'
        converges, but the precision of the extracted profile is lower.
    max_iter :: int
        The number of iterations to repeat the Horne extraction algorithm
        for. The cosmic ray masking has been removed, so the only benefit
        from increasing 'max_iter' is the potential to get a better
        constraint on the spatial profile.
    repeat :: bool
        Whether to repeat the spatial profile generation once an initial
        pass has been made. When your data is particularly noisy, it is
        helpful to keep this as 'True'.
    debug :: bool
        Allows for optional plotting.
    progress :: bool
        Enables a progress bar to be displayed.

    Returns:
    --------
    flux :: np.ndarray
        An array containing the extracted flux for each exposure.
    flux_err :: np.ndarray
        An array containing the extracted error for each exposure.
    """

    # Converts 2D arrays to 3D arrays
    original_shape = images.shape
    if len(original_shape) == 2:
        images = np.array([images])
        backgrounds = np.array([backgrounds])

    # Initializes several useful arrays
    N_images = len(images)
    N_wavelengths = len(images[0][0])
    flux = np.zeros((N_wavelengths, N_images))
    flux_err = np.zeros((N_wavelengths, N_images))

    # Iterates over every image
    for idx in tqdm(
        range(N_images), desc="Performing Optimal Extraction", disable=not progress
    ):

        # Creates initial spectral extraction / variance
        D = (images + backgrounds)[idx]
        S = backgrounds[idx]
        V = RN**2 + D

        # Initializes flux using median to mitigate cosmic rays
        f = np.median(images + backgrounds, axis=0)

        step = 0

        # Iterates until erroneous pixels have been flagged and removed
        while step < max_iter:

            # Generates new spatial profile and variance estimate
            P = generate_spatial_profile(
                (D - S) / f,
                bin_size=bin_size,
                profile=profile,
                profile_order=profile_order,
                repeat=repeat,
                debug=False,
            )

            V = RN**2 + np.abs(f * P + S)
            V = np.clip(V, 1e-20, None)

            # Re-calculates flux and variance using updated arrays
            numerator = np.sum(P * (D - S) / V, axis=0)
            denominator = np.sum(P**2 / V, axis=0)
            f = numerator / denominator
            f_var = np.sum(P, axis=0) / denominator

            step += 1

        flux[:, idx] = f
        flux_err[:, idx] = np.sqrt(f_var)

    flux = flux.T
    flux_err = flux_err.T

    if debug:
        pixel_positions = np.array(range(len(flux[0])))

        plt.rcParams["figure.figsize"] = (12, 4)
        plt.errorbar(
            pixel_positions,
            flux[0],
            yerr=flux_err[0],
            color="k",
            label="First Exposure",
            fmt="none",
        )
        plt.plot(
            pixel_positions,
            np.median(flux, axis=0),
            color="salmon",
            label="Median Exposure",
            zorder=-999,
        )
        plt.xlim(np.min(pixel_positions), np.max(pixel_positions))
        plt.xlabel("Pixel Position (Dispersion Axis)")
        plt.ylabel("Extracted Flux / Pixel")
        plt.legend()
        plt.show()

    return flux, flux_err


def trace_fit(
    image: np.ndarray, bin: int = 16, trace_order: int = 2, debug: bool = False
):
    """
    Fits a trace to a signal across the horizontal
    axis of an image. This is done by rebinning a
    user-given image, fitting a gaussian to each
    rebinned column, and fitting an n-dimensional
    curve to these gaussian positions.

    Parameters:
    -----------
    image :: np.ndarray
        Image with a signal spanning the horizontal
        axis of the detector.
    bin :: int
        Number of pixels to group into a single bin.
        Must be an integer multiple of the horizontal
        pixel count.
    trace_order :: int
        Order of the polynomial to be fit to our
        trace fit data.
    debug :: bool
        Allows plot generation.

    Returns:
    --------
    xpoints :: np.ndarray
        Horizontal pixel positions corresponding
        to our detected trace fit. This has been
        converted from the downsampled x-values
        to the original image x-values.
    locs :: np.ndarray
        Vertical locations of the detected trace
        positions.
    stds :: np.ndarray
        Standard deviations associated with each
        gaussian fit in the downsampled image.
    p_center :: np.poly1d
        Polynomial fit that traces our signal
        out across the detector.
    """

    # Rebins user-given image
    rebinned_image = rebin_image_columns(image, bin)

    # Defines trace data arrays
    locs = np.array([])
    stds = np.array([])

    # Iterates over each column in rebinned image
    for i in range(len(rebinned_image[0])):

        # Pulls brightness data for each column
        x_data = range(len(rebinned_image))
        y_data = list(rebinned_image[:, i])

        # Guesses that the parameters of our column Gaussian fit
        initial_guess = [max(y_data), y_data.index(max(y_data)), 1]

        # Fit Gaussian to data
        popt, pcov = curve_fit(_gaussian, x_data, y_data, p0=initial_guess)

        # Extract fitted parameters
        A_fit, mu_fit, sigma_fit = popt

        # Appends fit parameters to lists
        locs = np.append(locs, mu_fit)
        stds = np.append(stds, sigma_fit)

    # Rescales x_points to fit our unbinned image
    xpoints = bin * np.array(range(len(rebinned_image[0]))) + bin / 2

    # Creates a model for our trace
    z_center = np.polyfit(xpoints, locs, trace_order)
    p_center = np.poly1d(z_center)

    # Plotting
    if debug:

        # Plots rebinned image
        plt.rcParams["figure.figsize"] = (12, 4)
        plt.imshow(
            np.abs(rebinned_image),
            cmap="inferno",
            aspect="auto",
            norm="log",
            interpolation="none",
        )
        plt.colorbar(label="Pixel Counts")

        # Plots extracted position data along signal
        ds_xs = np.array(range(len(rebinned_image[0])))
        plt.scatter(ds_xs, locs, color="k")
        plt.errorbar(
            ds_xs,
            locs,
            yerr=stds,
            fmt="none",
            capsize=3,
            color="k",
            label="Signal Gaussian Position",
        )

        # Formatting
        plt.title(f"Rebinned Image (1 bin = {bin} pixels)")
        plt.legend()
        plt.show()

    return xpoints, locs, stds, p_center
