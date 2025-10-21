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
        "moffat": [_moffat, [0.5, -1, 5], [[0, 0, 4], [1, len(image), 20]]],
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

                popt, pcov = curve_fit(profile_function, rows, y, p0=p0, bounds=bounds)
                parameters.append(popt)
                successful_cols.append(cols_binned[idx])

                if debug:
                    plt.rcParams["figure.figsize"] = (12, 1)
                    plt.title(popt)
                    plt.scatter(rows, y)
                    plt.plot(rows, profile_function(rows, *popt))
                    plt.show()

            except:
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


def horne_extraction(
    images: np.ndarray,
    backgrounds: np.ndarray,
    profile: str = "gaussian",
    profile_order: int = 7,
    RN: float = 6.0,
    bin_size: int = 8,
    sigma_clip: float = 25.0,
    max_iter: int = 10,
    repeat: bool = True,
    debug: bool = False,
    update: bool = False,
):

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
        range(N_images), desc="Performing Optimal Extraction", disable=not update
    ):

        # Creates initial spectral extraction / variance
        D = (images + backgrounds)[idx]
        S = backgrounds[idx]
        V = RN**2 + D

        # Initializes flux using median to mitigate cosmic rays
        f = np.sum(D - S, axis=0)

        # Creates two arrays to stop running when flagged outliers match
        M = np.ones(D.shape)
        last_M = np.zeros(D.shape)

        step = 0

        # Iterates until erroneous pixels have been flagged and removed
        while not np.array_equal(M, last_M) and step < max_iter:

            last_M = M.copy()

            # Generates new spatial profile and variance estimate
            P = generate_spatial_profile(
                D - S,
                bin_size=bin_size,
                profile=profile,
                profile_order=profile_order,
                repeat=repeat,
                debug=debug,
            )
            P *= M

            V = RN**2 + np.abs(f * P + S)
            V = np.clip(V, 1e-20, None)

            # Removes the brightest pixel if it exceeds clipping threshold
            flagging_threshold = M * np.sqrt(((D - S - f * P) ** 2 / V))
            if np.max(flagging_threshold > sigma_clip):
                M[flagging_threshold == np.max(flagging_threshold)] = 0

            # Re-calculates flux and variance using updated arrays
            numerator = np.sum(M * P * (D - S) / V, axis=0)
            denominator = np.sum(M * P**2 / V, axis=0)
            f = numerator / denominator
            f_var = np.sum(M * P, axis=0) / denominator

            step += 1

        flux[:, idx] = f
        flux_err[:, idx] = np.sqrt(f_var)

    return flux, flux_err


def extract_flux(
    images: np.ndarray,
    images_bg: np.ndarray,
    bin_size: float = 2**6,
    trace_order: int = 2,
    model: str = "fit moving",
    extraction_width: int = 10,
    box_pos: float = 0.0,
    debug: bool = False,
    give_fit_params: bool = False,
    show_param_evolution: bool = False,
    update: bool = False,
):
    """
    Extracts flux from a given image using a
    specified extraction technique. The user
    must provide at least an image for extraction
    w/ the same image before background subtraction.

    Parameters:
    -----------
    images :: np.ndarray
        Calibrated images from which a signal will be
        extracted. This function assumes that this image
        has been background-subtracted.
    images_bg :: np.ndarray
        Identical image to the first one, but before
        the background subtraction has been applied.
    trace_order :: int
        Order of the polynomial used to fit to the
        detected signal locations.
    model :: str
        Phrase indicating what type of extraction technique
        to use. Valid options are...

                   'boxcar' ~ Straight line trace at given position
             'fit averaged' ~ Fits one trace to the average of
                               all user-given images
               'fit moving' ~ Fits a trace to each individual image.
                              Most accurate, but prone to failure if
                              a Gaussian curve poorly describes your
                              given data.

    extraction_width :: int
        Number of pixels above (and below) the trace
        to use for flux extraction.
    box_pos :: float
        The y-position at which your boxcar trace should
        be centered. Only required if using the 'boxcar'
        model.
    debug :: bool
        Allows plot generation.
    give_fit_params :: bool
        Allows for trace parameters to be returned from
        the function.
    show_param_evolution :: bool
        Allows for the plotting of trace fit parameters.

    Returns:
    --------
    spectra :: np.ndarray
        Array of spectra extracted from the user-given
        images. If multiple images are provided, each
        row represents a single spectrum.
    errs :: np.ndarray
        Array of errors associated with the extracted
        spectra. Follows the same structure as the
        spectra array.
    fit_coefficients :: np.ndarray
        Array of trace fit parameters for every image.
        This is an OPTIONAL return, and follows the
        same convention as the above two arrays.
    """

    # List for storing trace fit coefficients
    fit_coefficients = []

    # Fixes formatting for single images
    if len(images.shape) == 2:
        images = np.array([images])
    if len(images_bg.shape) == 2:
        images_bg = np.array([images_bg])

    # Initializes arrays for spectral data
    spectra = np.zeros((len(images), len(images[0][0])))
    errs = np.zeros((len(images), len(images[0][0])))

    # Fits one trace to average image
    if model == "fit averaged":

        # Fits for an n-dimensional trace to averaged image
        avg_im = np.median(images, axis=0)
        xs, locs, stds, p = trace_fit(
            avg_im, bin=bin_size, trace_order=trace_order, debug=debug
        )
        fit_coefficients = p.coefficients

    # Uses a contant boxcar trace
    if model == "boxcar":
        xs = np.linspace(0, len(images[0][0]), 20)
        locs = np.array([box_pos for _ in xs])
        stds = np.array([0 for _ in xs])
        p = np.poly1d([box_pos])

    # Iterates over every image
    for idx in tqdm(range(len(images)), desc="extracting flux", disable=not update):

        # Extracts individual images
        im = images[idx]
        bg = images_bg[idx]

        # Fits trace for every image
        if model == "fit moving":

            # Fits for an n-dimensional trace to
            xs, locs, stds, p = trace_fit(
                im, bin=bin_size, trace_order=trace_order, debug=debug
            )
            fit_coefficients.append(p.coefficients)

        # Performs flux extraction
        _, flux, err = apply_trace_extraction(
            im,
            bg,
            p,
            xpoints=xs,
            locs=locs,
            stds=stds,
            debug=debug,
            N_pix=extraction_width,
        )

        # Adds flux and error data to arrays
        spectra[idx] = flux
        errs[idx] = err

    # Allows for trace parameter return
    if give_fit_params:

        # Displays fit parameters evolution
        if model == "fit moving" and show_param_evolution:

            # Creates array of image indexes for plotting
            idxs = np.array(range(len(fit_coefficients)))

            # Iterates over each trace coefficient
            for i in range(len(fit_coefficients[0])):

                # Plots evolution of trace parameter over images
                plt.rcParams["figure.figsize"] = (10, 3)
                plt.scatter(idxs, np.array(fit_coefficients)[:, i], color="k")
                plt.title(f"Coefficient Order: {len(fit_coefficients[0])-1-i}")
                plt.xlabel("Image Index")
                plt.ylabel("Value")
                plt.show()

        return spectra, errs, np.array(fit_coefficients)

    return spectra, errs


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


def apply_trace_extraction(
    image,
    image_bg,
    p_trace,
    xpoints=None,
    locs=None,
    stds=None,
    N_pix=5,
    RN=6.0,
    debug=False,
):
    """
    Applies a defined trace to a user-given image
    to extract signal flux within a given range of
    pixels. Some arguments are only required if the
    user intends to use the debugging plots. This
    assumed the user-given images are already in units
    of photons (not in ADU).

    Parameters:
    -----------
    image :: np.ndarray
        Image that the trace model describes
        that flux will be extracted from.
    image_bg :: np.ndarray
        The same image we are extracting flux
        from, but before background corrections
        have been applied.
    p_trace :: np.poly1d
        Polynomial model describing the trace
        that was fit to the provided image.
    xpoints :: np.ndarray
        X-positions of each point extracted from
        the signal (only required for debug plots)
    locs :: np.ndarray
        Y-positions of each point extracted from
        the signal (only required for debug plots)
    stds :: np.ndarray
        Standard deviations of each point extracted
        the signal (only required for debug plots)
    N_pix :: int
        Number of pixels above (and below) trace to
        extract.
    RN :: float
        Read noise associated with image. This is
        used for calculating the error of our
        extractions.
    debug :: bool
        Allows plot generation.

    Returns:
    --------
    xs :: np.ndarray()
        Horizontal pixel positions corresponding to
        each extracted flux value. This is NOT in
        wavelength-calibrated units.
    extracted_flux :: np.ndarray
        Extracted flux across our extraction aperture.
    extracted_error :: np.ndarray
        Error associated with the extracted flux.
    """

    xs = range(len(image[0]))

    lower_bounds = [round(lower) - N_pix for lower in p_trace(range(len(image[0])))]
    upper_bounds = [round(upper) + N_pix for upper in p_trace(range(len(image[0])))]

    # Finds the error squared of our original image
    err_im_squared = image_bg + RN**2

    # Extracts flux and error from our defined aperture
    extracted_flux = np.array(
        [np.sum(image[i:j, k]) for i, j, k in zip(lower_bounds, upper_bounds, xs)]
    )
    extracted_error = np.array(
        [
            np.sqrt(np.sum(err_im_squared[i:j, k]))
            for i, j, k in zip(lower_bounds, upper_bounds, xs)
        ]
    )

    if debug:

        # Plots original image
        plt.rcParams["figure.figsize"] = (12, 4)
        plt.imshow(
            np.abs(image),
            cmap="inferno",
            aspect="auto",
            norm="log",
            interpolation="none",
        )
        plt.colorbar(label="Pixel Counts")

        # Plots trace fits
        plt.plot(xs, p_trace(xs), color="k", linewidth=3, alpha=0.5, label="Trace Fit")
        plt.plot(
            xs,
            upper_bounds,
            color="blue",
            linewidth=3,
            label=f"Extraction Region ({N_pix} Pixels)",
        )
        plt.plot(xs, lower_bounds, color="blue", linewidth=3)

        # Plots extracted position data along signal
        plt.scatter(xpoints, locs, color="k")
        plt.errorbar(
            xpoints,
            locs,
            yerr=stds,
            fmt="none",
            capsize=3,
            color="k",
            label="Signal Gaussian Position",
        )

        # Formatting
        plt.title("Flux Extraction Aperture on Original Image")
        plt.xlim(0, len(image[0]))
        plt.legend(loc="upper left", shadow=True)
        plt.show()

        # Plots extracted flux
        plt.rcParams["figure.figsize"] = (10, 5)
        plt.scatter(xs, extracted_flux, color="k", s=0.1)
        plt.errorbar(
            xs, extracted_flux, yerr=extracted_error, fmt="none", capsize=2, color="k"
        )
        plt.xlim(0, len(xs))
        plt.xlabel("Column (pixels)")
        plt.ylabel("Extracted Flux")
        plt.show()

    return np.array(xs), np.array(extracted_flux), np.array(extracted_error)
