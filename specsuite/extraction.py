import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import warnings
from importlib.resources import files

from tqdm import tqdm
from astropy.stats import mad_std
from scipy.optimize import curve_fit
from .utils import (
    _gaussian,
    _moffat,
    rebin_image_columns,
    convolve_to_resolution,
    plot_image,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


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

            V = RN**2 + np.abs(f * P.copy() + S)
            V[V < 1e-20] = 0
            # V = np.clip(V, 1e-20, None)

            # Re-calculates flux and variance using updated arrays
            numerator = np.sum(P.copy() * (D - S) / V.copy(), axis=0)
            denominator = np.sum(P.copy() ** 2 / V.copy(), axis=0)

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


class ExtinctionModel:
    def __init__(
        self, throughput_model, rs_tau0, o2_abundance, humidity, loss_constant, R
    ):

        # Stores the parameters of the extinction model as attributes of the class
        self.throughput_model = throughput_model
        self.rs_tau0 = rs_tau0
        self.o2_abundance = o2_abundance
        self.humidity = humidity
        self.loss_constant = loss_constant
        self.R = R

    def __call__(self, airmass):
        return np.exp(np.log(self.throughput_model) * airmass)


def _load_extinction_file(name: str):
    path = files("specsuite.extinction_models") / f"{name}.npy"
    return np.load(path)


def estimate_extinction_coefficients(
    airmass: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    max_iterations: int = 5,
    clip_threshold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Attempts to estimate the extinction coefficients (taus) over the
    course of a single night of observations. The provided data arrays
    should correspond to a comparison star that should be relatively
    non-variable over the course of the night.

    Parameters:
    -----------
    airmass :: np.ndarray
        Airmass values corresponding to each observation in the flux array.
        Plane-parallel estimates are accurate enough!
    flux :: np.ndarray
        2D array of shape (n_observations, n_wavelength_bins) containing the
        flux measurements for the comparison star at each wavelength bin and
        observation time.
    error :: np.ndarray
        2D array of shape (n_observations, n_wavelength_bins) containing the
        uncertainties associated with each flux measurement in the flux array.
    max_iterations :: int
        Maximum number of iterations used for the sigma-clipping process.
    clip_threshold :: float
        The threshold (in units of median absolute deviation) beyond which
        a data point is considered an outlier and excluded from the fit.

    Returns:
    --------
    taus :: np.ndarray
        1D array of estimated extinction coefficients (taus) for each wavelength bin.
    taus_error :: np.ndarray
        1D array of uncertainties associated with each estimated extinction coefficient.
    """

    assert (max_iterations > 0) and isinstance(
        max_iterations, int
    ), "'max_iterations' must be a positive integer!"
    assert clip_threshold > 0, "'clip_threshold' must be positive!"

    # Since '.shape' is used, this ensures provided data are Numpy arrays
    flux = np.array(flux)
    error = np.array(error)
    airmass = np.array(airmass)

    assert flux.shape == error.shape, "Flux and error arrays must have the same shape!"
    assert flux.shape[0] == len(
        airmass
    ), "Number of observations in flux must match length of airmass array!"

    # Initialize empty arrays
    taus = np.array([])
    taus_error = np.array([])

    # Iterate over each wavelength bin in the flux array
    for wav_bin in range(len(flux[0])):

        # Create a mask to exclude NaN values in the flux array
        mask = np.ones_like(airmass, dtype=bool)
        mask[np.where(np.isnan(flux.T[wav_bin]))] = False

        # Use a try-except block to handle potential errors during the fitting process
        try:

            # Can run less that 'max_iterations' if fit converges
            for i in range(max_iterations):

                x = airmass
                y = np.log(flux.T[wav_bin])
                w = np.abs(flux.T[wav_bin]) / error.T[wav_bin]

                # Perform a weighted linear fit
                best_fit, best_cov = np.polyfit(
                    x[mask], y[mask], deg=1, w=w[mask], cov=True
                )
                param_error = np.sqrt(np.diag(best_cov))

                # Calculate residuals and normalize by the median absolute deviation
                residuals = (y - np.poly1d(best_fit)(x)) / w
                residuals /= mad_std(residuals)

                # If any residuals exceed the clip threshold, update mask
                if np.any(np.abs(residuals[mask]) > clip_threshold):
                    mask_value = np.max(np.abs(residuals[mask]))
                    mask[np.where(np.abs(residuals) == mask_value)] = False
                else:
                    break

            taus = np.append(taus, best_fit[0])
            taus_error = np.append(taus_error, param_error[0])

        # If the fitting process fails, append 0 to the taus and taus_error lists
        except ValueError:
            taus = np.append(taus, 0)
            taus_error = np.append(taus_error, 0)

    return taus, taus_error


def generate_extinction_model(
    wavelengths: np.ndarray,
    rs_tau0: float = 0.0,
    o2_abundance: float = 20000,
    humidity: float = 50,
    w_offset: float = 0.0,
    loss_constant: float = 1.0,
    R: float = 2000,
    airmass: float = 1.0,
) -> np.ndarray:
    """
    Generates a model for telluric extinction with individual contributions
    from Rayleigh scattering, O2 absorption, and H2O absorption. There is
    also a 'constant loss' term to account for achromatic losses, a
    wavelength offset to account for calibration errors, and a resolution
    parameter to apply convolution to the model.

    Parameters:
    -----------
    wavelengths :: np.ndarray
        A 1D array of wavelengths at which to evaluate the extinction model.
        Ideally, this should have some astropy units attached, but if not
        will assume Angstroms.
    rs_tau0 :: float
        The unitless tau0 parameter for Rayleigh scattering, which sets the
        overall strength of the Rayleigh scattering contribution.
    o2_abundance :: float
        The abundance of O2 in the atmosphere, which scales the O2 absorption
        model. Typically, a value around 100000-300000 is reasonable for
        Earth's atmosphere.
    humidity :: float
        The relative humidity percentage, which scales the H2O absorption model.
        Only values 0 <= humidity <= 100 are physically meaningful.
    w_offset :: float
        A wavelength offset in the same units as the input wavelengths, which
        accounts for a linear shift in the wavelength solution.
    loss_constant :: float
        A constant multiplicative factor between 0 and 1 that accounts for
        achromatic losses such as clouds or instrumental throughput issues.
    R :: float
        The spectral resolution (lambda/delta_lambda) to which the model should
        be convolved. Uses an FFT convolution for runtime efficiency.
    airmass :: float
        The airmass of the observation, which scales all extinction components.

    Returns:
    --------
    complete_model :: np.ndarray
         The combined extinction model evaluated at the input wavelengths.
    """

    # Performing a single check here prevents separate conversions for each model
    if isinstance(wavelengths, u.Quantity):
        wavelengths = wavelengths.to(u.AA).value

    # Same logic as above, but for the wavelength offset parameter
    if isinstance(w_offset, u.Quantity):
        w_offset = w_offset.to(u.AA).value

    # Prevents unphysical parameters from being used in the model
    assert R > 0, "Resolution must be positive!"
    assert 0 <= humidity <= 100, "Humidity must be between 0% and 100%!"
    assert rs_tau0 >= 0, "Rayleigh scattering tau0 cannot be negative!"
    assert o2_abundance >= 0, "O2 abundance cannot be negative!"
    assert airmass >= 1, "Airmass must be greater than or equal to 0!"
    assert 0 <= loss_constant <= 1, "Loss constant must be between 0 and 1!"

    # Loads loss rate models + wavelengths from package files
    o2_model = _load_extinction_file("o2_array")
    h2o_model = _load_extinction_file("humidity_array")
    model_wavelengths = _load_extinction_file("model_wavelengths")

    # Restrict to the wavelength range of the data
    wmin, wmax = np.min(model_wavelengths), np.max(model_wavelengths)
    mask = (model_wavelengths >= wmin) & (model_wavelengths <= wmax)

    # Masks out wavelengths outside the data range + clips NaNs in models s
    o2_model = np.nan_to_num(o2_model[mask], nan=0.0)
    h2o_model = np.nan_to_num(h2o_model[mask], nan=0.0)
    model_wavelengths = model_wavelengths[mask]

    # Scales models by abundance / rate and airmass
    h2o_model *= humidity * airmass
    o2_model *= o2_abundance * airmass
    rayleigh_model = -rs_tau0 * (5000 / model_wavelengths) ** 4 * airmass

    #
    complete_model = loss_constant * np.exp(o2_model + h2o_model + rayleigh_model)

    # Hopefully no NaNs make it this far, but this acts as a final safety
    complete_model = np.nan_to_num(complete_model, nan=1.0, posinf=1.0, neginf=0.0)
    complete_model = convolve_to_resolution(model_wavelengths, complete_model, R)

    # Applies convolution to the model and interpolates to the data wavelengths
    complete_model = np.interp(
        x=wavelengths,
        xp=model_wavelengths + w_offset,
        fp=complete_model,
    )

    return complete_model


def fit_extinction_model(
    wavelengths: np.ndarray,
    taus: np.ndarray,
    taus_error: np.ndarray,
    p0: dict = None,
    bounds: dict = None,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits a multi-component telluric extinction model to the input data.
    This model includes molecular absorption from O2 and H2O, Rayleigh
    scattering, and an achromatic loss term. The model is convolved to a
    specified resolution and includes a wavelength offset parameter to
    account for calibration errors.

    Parameters:
    -----------
    wavelengths :: np.ndarray
        A 1D array of wavelengths at which the extinction model should be
        evaluted. Ideally, this should have some astropy units attached,
        but if not will assume Angstroms.
    taus :: np.ndarray
        A 1D array of the same length as wavelengths, containing the
        estimated effective optical depth (tau) of the atmosphere at each
        wavelength.
    taus_error :: np.ndarray
        A 1D array of the same length as wavelengths, containing the
        uncertainties on the tau estimates.
    p0 :: dict
        A dictionary containing the initial guess for the model parameters.
        The expected keys are: 'rs_tau0', 'o2_abundance', 'humidity', 'w_offset',
        'loss_constant', and 'R'. If None, default values will be used.
    bounds :: dict
        A dictionary containing the bounds for the model parameters during fitting.
        The expected keys are the same as for p0, and the values should be tuples
        specifying the (min, max) bounds for each parameter. If None, default bounds
        will be used.
    debug :: bool
        Optional plotting.

    Returns:
    --------
    best_model :: np.ndarray
        The best-fit telluric throughput model evaluated at the input wavelengths.
    fitted_values :: np.ndarray
        The best-fit values for the model parameters in the following order:
           - rs_tau0
           - o2_abundance
           - humidity
           - w_offset
           - loss_constant
           - R
    fitted_values_errors :: np.ndarray
        The 1-sigma uncertainties on the fitted parameters, calculated from the
        covariance matrix returned by curve_fit.
    """

    # Using dictionaries to make it easier to know which variables are which
    if p0 is None:
        p0 = {
            "rs_tau0": 0.1,
            "o2_abundance": 30000,
            "humidity": 50,
            "w_offset": 0.0,
            "loss_constant": 1.0,
            "R": 1000,
        }
    if bounds is None:
        bounds = {
            "rs_tau0": (0.0, np.inf),
            "o2_abundance": (0.0, np.inf),
            "humidity": (0.0, 100.0),
            "w_offset": (-100.0, 100.0),
            "loss_constant": (0.0, 1.0),
            "R": (1.0, np.inf),
        }

    # Should only fail if user provides an invalid p0 or bounds
    assert all(
        key in p0
        for key in [
            "rs_tau0",
            "o2_abundance",
            "humidity",
            "w_offset",
            "loss_constant",
            "R",
        ]
    ), (
        "p0 must contain keys: "
        + "'rs_tau0', 'o2_abundance', 'humidity', 'w_offset', 'loss_constant', 'R'"
    )
    assert all(
        key in bounds
        for key in [
            "rs_tau0",
            "o2_abundance",
            "humidity",
            "w_offset",
            "loss_constant",
            "R",
        ]
    ), (
        "bounds must contain keys: "
        + "'rs_tau0', 'o2_abundance', 'humidity', 'w_offset', 'loss_constant', 'R'"
    )

    # Convert p0 and bounds to the format expected by curve_fit
    p0_values = [
        p0["rs_tau0"],
        p0["o2_abundance"],
        p0["humidity"],
        p0["w_offset"],
        p0["loss_constant"],
        p0["R"],
    ]
    bounds_values = (
        [
            bounds["rs_tau0"][0],
            bounds["o2_abundance"][0],
            bounds["humidity"][0],
            bounds["w_offset"][0],
            bounds["loss_constant"][0],
            bounds["R"][0],
        ],
        [
            bounds["rs_tau0"][1],
            bounds["o2_abundance"][1],
            bounds["humidity"][1],
            bounds["w_offset"][1],
            bounds["loss_constant"][1],
            bounds["R"][1],
        ],
    )

    # Remove any NaN values from the data to avoid issues with curve_fit
    if np.any(np.isnan(taus)):
        mask = ~np.isnan(taus)
        wavelengths = wavelengths[mask]
        taus = taus[mask]
        taus_error = taus_error[mask]

    # Fits a telluric throughput model to the data
    fitted_values, pcov = curve_fit(
        generate_extinction_model,
        wavelengths,
        np.exp(taus),
        sigma=taus_error,
        absolute_sigma=True,
        p0=p0_values,
        bounds=bounds_values,
    )

    # Calculate the errors on the fitted parameters from the covariance matrix
    fitted_values_errors = np.sqrt(np.diag(pcov))

    # Generate the best-fit model using the fitted parameters
    best_model = generate_extinction_model(
        wavelengths=wavelengths,
        rs_tau0=fitted_values[0],
        o2_abundance=fitted_values[1],
        humidity=fitted_values[2],
        w_offset=fitted_values[3],
        loss_constant=fitted_values[4],
        R=fitted_values[5],
    )

    # Optional debugging plots
    if debug:

        # Generate the individual components of the model for visualization
        o2_component = generate_extinction_model(
            wavelengths=wavelengths,
            rs_tau0=0.0,
            o2_abundance=fitted_values[1],
            humidity=0.0,
            w_offset=0.0,
            loss_constant=1.0,
            R=fitted_values[5],
        )
        humidity_component = generate_extinction_model(
            wavelengths=wavelengths,
            rs_tau0=0.0,
            o2_abundance=0.0,
            humidity=fitted_values[2],
            w_offset=0.0,
            loss_constant=1.0,
            R=fitted_values[5],
        )
        rayleigh_component = generate_extinction_model(
            wavelengths=wavelengths,
            rs_tau0=fitted_values[0],
            o2_abundance=0.0,
            humidity=0.0,
            w_offset=0.0,
            loss_constant=fitted_values[4],
            R=fitted_values[5],
        )

        # Sets a reasonable y-scale for seeing the data + composite model
        data_min = 0.5
        if np.min(np.exp(taus)) < 0.5:
            data_min = 0.0

        # Sets a reasonable y-scale for seeing the individual components
        component_min = np.min([o2_component, humidity_component, rayleigh_component])
        component_min = np.max([component_min - 0.05, 0.0])

        # Creates a 4-panel plot showing the data + model and the individual components
        _, axs = plt.subplots(
            4,
            1,
            figsize=(8, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1, 1, 1]},
        )

        # Raw data + best-fit model
        axs[0].errorbar(
            wavelengths,
            np.exp(taus),
            yerr=taus_error,
            markersize=3,
            capsize=3,
            color="k",
            alpha=0.9,
        )
        axs[0].plot(wavelengths, best_model, color="red", zorder=999)
        axs[0].set_ylabel("Atmospheric Transmission")
        axs[0].set_xlim(wavelengths[0], wavelengths[-1])
        axs[0].set_ylim(data_min, 1.0)

        # O2 absorption component
        axs[1].plot(wavelengths, o2_component, color="blue")
        axs[1].fill_between(wavelengths, 1.1, o2_component, color="blue", alpha=0.3)
        axs[1].set_ylim(component_min, 1.0)
        axs[1].set_yticks([])
        axs[1].text(
            0.01,
            0.2,
            "O2 Molecular Absorption",
            transform=axs[1].transAxes,
            fontsize=12,
            verticalalignment="center",
        )

        # H2O absorption component
        axs[2].plot(wavelengths, humidity_component, color="orange")
        axs[2].fill_between(
            wavelengths, 1.1, humidity_component, color="orange", alpha=0.3
        )
        axs[2].set_ylim(component_min, 1.0)
        axs[2].set_yticks([])
        axs[2].text(
            0.01,
            0.2,
            "H2O Molecular Absorption",
            transform=axs[2].transAxes,
            fontsize=12,
            verticalalignment="center",
        )

        # Rayleigh scattering + achromatic loss component
        axs[3].plot(wavelengths, rayleigh_component, color="green", label="Rayleigh")
        axs[3].fill_between(
            wavelengths, 1.1, rayleigh_component, color="green", alpha=0.3
        )
        axs[3].set_ylim(component_min, 1.0)
        axs[3].set_yticks([])
        axs[3].set_xlabel("Wavelength (Å)")
        axs[3].text(
            0.01,
            0.2,
            "Rayleigh + Achromatic Losses",
            transform=axs[3].transAxes,
            fontsize=12,
            verticalalignment="center",
        )

        plt.show()

    extinction_model = ExtinctionModel(
        throughput_model=best_model,
        rs_tau0=fitted_values[0],
        o2_abundance=fitted_values[1],
        humidity=fitted_values[2],
        loss_constant=fitted_values[4],
        R=fitted_values[5],
    )

    return extinction_model, fitted_values, fitted_values_errors


def _infer_flux_baselines(
    times: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    max_iterations: int = 5,
    outlier_threshold: int = 3,
    debug: bool = True,
) -> np.ndarray:
    """
    Approximates the baseline flux for each wavelength bin by fitting
    a sloped line to the flux for the bin over time. The output array
    is a 2D array of the same shape as 'flux' where each column is the
    flux / fitted baseline for that wavelength bin. This can be thought
    of as the ratio of the observed flux to the expected baseline.

    Parameters:
    -----------
    times :: np.ndarray
        The times at which each exposure was taken, should correspond to
        the rows of 'flux' and 'error'. There should be no astropy units
        associated with this array.
    flux :: np.ndarray
        A 2D array of flux values with shape (N_exposures, N_wavelength_bins).
    error :: np.ndarray
        A 2D array of flux uncertainties with the same shape as 'flux'.
    max_iterations :: int
        The maximum number of iterations for sigma-clipping to remove
        outliers when fitting the baseline. Should be a positive integer.
    outlier_threshold :: int
        The number of MADs above which a point is considered an outlier
        when fitting the baseline. Should be a positive integer.
    debug :: bool
        Enables optional debugging plots.

    Returns:
    --------
    ratios :: np.ndarray
        The ratio of the observed flux to the fitted baseline for each wavelength bin,
        with the same shape as 'flux'.
    """

    # Pre-allocate arrays for speed and memory efficiency
    ratios = np.zeros(flux.shape)
    composite_mask = np.ones(flux.shape, dtype=bool)

    # Loop over each wavelength bin to find the baseline flux
    for ref_idx in range(len(flux[0])):

        # Sigma-clipping to try and remove cloud-affected exopsures for fitting
        mask = np.ones(times.shape, dtype=bool)

        # Should run at least once, but will stop early if no outliers are found
        for _ in range(max_iterations):

            # Fits a sloped baseline to the flux at this wavelength bin
            coeffs = np.polyfit(
                times[mask],
                flux[:, ref_idx][mask],
                w=1 / error[:, ref_idx][mask],
                deg=1,
            )
            p_baseline = np.poly1d(coeffs)

            # Outliers are defined as a multiple of the MAD from the residuals
            residuals = flux[:, ref_idx] - p_baseline(times)
            outliers = np.abs(residuals) > outlier_threshold * mad_std(residuals[mask])
            mask[outliers] = False

        # Since this is a fast operation, no need to require 'debug'=True
        composite_mask[:, ref_idx] = mask

        # In-place operation to save memory
        ratios[:, ref_idx] = flux[:, ref_idx] / p_baseline(times)

    # Optional debugging plots
    if debug:

        # Nearly all wavelengths for a given exposure should be masked
        plot_image(
            composite_mask,
            xlabel="Wavelength Bin",
            ylabel="Exposure Index",
            cbar_label=None,
            title="Outlier Mask for Flux Baseline Fitting",
        )

    return ratios


def _infer_achromatic_deviations(
    ratios: np.ndarray,
    max_iterations: int = 5,
    outlier_threshold: int = 3,
    debug: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Infers the achromatic deviations in flux for each wavelength bin by
    fitting a constant offset to the 'ratios' output from
    '_infer_flux_baselines'. The output is a 2D array of the same shape
    as 'flux' where each column is the flux / fitted baseline for that
    wavelength bin. This can be thought of as the ratio of the observed
    flux to the expected baseline, with the wavelength-dependent trends
    removed.

    Parameters:
    -----------
    ratios :: np.ndarray
        The output from '_infer_flux_baselines', which is the ratio of the
        observed flux to the fitted baseline for each wavelength bin.
    max_iterations :: int
        The maximum number of iterations for sigma-clipping to remove
        outliers when fitting the achromatic deviations. Should be a
        positive integer.
    outlier_threshold :: int
        The number of MADs above which a point is considered an outlier
        when fitting the achromatic deviations. Should be a positive integer.
    debug :: bool
        Enables optional debugging plots.

    Returns:
    --------
    achromatic_bump :: np.ndarray
        A 2D array with the same shape as 'flux' and 'error' that contains
        the correction factor to remove achromatic bumps.
    achromatic_bump_error :: np.ndarray
        A 2D array with the same shape as 'flux' and 'error' that contains
        the uncertainty on the correction factor to remove achromatic bumps.
    """

    # Pre-allocate arrays for speed and memory efficiency
    achromatic_bump = np.zeros(ratios.shape)
    achromatic_bump_error = np.zeros(ratios.shape)
    composite_mask = np.ones(ratios.shape, dtype=bool)

    # Loop over each wavelength bin to find the achromatic deviations
    for idx in range(ratios.shape[0]):

        # Sigma-clipping to try and remove wavelength-dependent outliers
        mask = np.ones(ratios.shape[1], dtype=bool)

        # Should run at least once, but will stop early if no outliers are found
        for _ in range(max_iterations):

            # Outliers are defined as a multiple of the MAD from the residuals
            average = np.nanmedian(ratios[idx][mask])
            mad = mad_std(ratios[idx][mask])
            outliers = np.abs(ratios[idx] - average) > outlier_threshold * mad
            mask[outliers] = False

        # Since this is a fast operation, no need to require 'debug'=True
        composite_mask[idx] = mask

        # In-place operation to save memory
        achromatic_bump[idx] = 1 - np.nanmedian(ratios[idx][mask])
        achromatic_bump_error[idx] = mad_std(ratios[idx][mask])

    if debug:

        # Nearly all exposures for a given wavelength bin should be masked
        plot_image(
            composite_mask.T,
            xlabel="Exposure Index",
            ylabel="Wavelength Bin",
            cbar_label=None,
            title="Outlier Mask for Wavelength-Independent Deviation Fitting",
        )

    return achromatic_bump, achromatic_bump_error


def fit_achromatic_bumps(
    times: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    airmasses: np.ndarray,
    throughput_model: ExtinctionModel,
    max_iterations: int = 5,
    outlier_threshold: int = 3,
    debug: bool = False,
) -> np.ndarray:
    """
    Parameters:
    -----------
    times :: np.ndarray
        The times at which each exposure was taken, should correspond to
        the rows of 'flux' and 'error'. There should be no astropy units
        associated with this array, but if there are, they will be
        removed internally.
    flux :: np.ndarray
        A 2D array of flux values with shape (N_exposures, N_wavelength_bins).
    error :: np.ndarray
        A 2D array of flux uncertainties with the same shape as 'flux'.
    airmasses :: np.ndarray
        A 1D array of airmass values for each exposure, should correspond
        to the rows of 'flux' and 'error'. There should be no astropy units
        associated with this array.
    throughput_model :: ExtinctionModel
        A model that takes in airmass values and outputs the expected throughput
        at each wavelength. This is used to correct the flux values before fitting
        the baseline, to try and remove the effects of telluric absorption.
    max_iterations :: int
        The maximum number of iterations for sigma-clipping routines to
        remove outliers. Should be a positive integer.
    outlier_threshold :: int
        The number of MADs above which a point is considered an outlier
        when fitting the baseline or achromatic deviations. Should be a
        positive integer.
    debug :: bool
        Enables optional debugging plots.

    Returns:
    --------
    achromatic_correction :: np.ndarray
        A 2D array with the same shape as 'flux' and 'error' that contains
        the multiplicative correction factor to remove achromatic bumps.
    """

    assert (
        times.shape == airmasses.shape
    ), "Times and airmasses must have the same shape!"
    assert (
        times.shape[0] == flux.shape[0]
    ), "Times and flux must have the same number of exposures!"
    assert flux.shape == error.shape, "Flux and error must have the same shape!"
    assert (max_iterations > 0) and isinstance(
        max_iterations, int
    ), "'max_iterations' must be a positive integer!"

    # Ensure times is a numpy array with no units
    if isinstance(times, u.Quantity):
        times_array = times.value.copy()
    else:
        times_array = np.array(times).copy()

    # Estimates the expected telluric throughput of each exposure
    throughput_image = np.array([throughput_model(X) for X in airmasses])

    flux_array = flux.copy() / throughput_image
    error_array = error.copy() / throughput_image

    # Infers the ratio between observed flux and a fitted baseline flux
    ratios = _infer_flux_baselines(
        times_array,
        flux_array,
        error_array,
        max_iterations=max_iterations,
        outlier_threshold=outlier_threshold,
        debug=debug,
    )

    # Infers constant offsets in flux caused by achromatic contaminants like clouds
    achromatic_bump, achromatic_bump_error = _infer_achromatic_deviations(
        ratios,
        max_iterations=max_iterations,
        outlier_threshold=outlier_threshold,
        debug=debug,
    )

    # Ensures that 'achromatic_correction' is a simple multiplicative correction
    achromatic_correction = 1 + achromatic_bump.copy()
    achromatic_correction_error = achromatic_bump_error.copy()

    # Optional debugging plots
    if debug:

        # Flux array copied to prevent accidentally overwriting the original flux values
        corrected_flux = flux_array * achromatic_correction

        _, axs = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
        vmin, vmax = np.percentile(corrected_flux, [1, 99])
        axs[0].imshow(
            flux_array,
            aspect="auto",
            origin="lower",
            cmap="inferno",
            interpolation="none",
            vmin=vmin,
            vmax=vmax,
        )
        axs[1].imshow(
            corrected_flux,
            aspect="auto",
            origin="lower",
            cmap="inferno",
            interpolation="none",
            vmin=vmin,
            vmax=vmax,
        )

        plt.show()

    return achromatic_correction, achromatic_correction_error


def estimate_atmospheric_loss(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    times: np.ndarray,
    airmass: np.ndarray,
    p0: dict = None,
    bounds: dict = None,
    max_iterations: int = 5,
    outlier_threshold: float = 5.0,
    return_components: bool = False,
    debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """
    Uses a multi-component modelling approach to estimate atmospheric
    losses for ground-based, time-series spectroscopic observations.

    The chromatic model includes airmass-dependent extinction from Rayleigh
    scattering, O2 absorption, H2O absorption, and an achromatic loss
    term to account for an average loss across all wavelengths. It also
    includes a wavelength offset parameter to account for calibration errors
    and a resolution parameter to apply convolution to the model. O2 and
    H2O extinction is estimated using pre-computed models based on the
    TelFit package.

    The achromatic model attempts to capture deviations in flux that happen
    sporadically across all wavelengths in any given exposure. This could be
    caused by clouds, instrumental issues, or any plausibly-achromatic source
    of light loss.

    Parameters:
    -----------
    wavelengths :: np.ndarray
        A 1D array of wavelengths at which the extinction model should be
        evaluated.
    flux :: np.ndarray
        A 2D array of flux values with shape (N_exposures, N_wavelength_bins).
        Should be on a linear scale (not magnitudes) and should not have
        any telluric correction applied.
    error :: np.ndarray
        A 2D array of flux uncertainties with the same shape as 'flux'.
    times :: np.ndarray
        The times at which each exposure was taken, should correspond to
        the rows of 'flux' and 'error'. There should be no astropy units
        associated with this array, but if there are, they will be removed
        internally.
    airmass :: np.ndarray
        A 1D array of airmass values for each exposure, should correspond
        to the rows of 'flux' and 'error'.
    p0 :: dict
        A dictionary containing the initial guess for the model parameters.
        The expected keys are: 'rs_tau0', 'o2_abundance', 'humidity',
        'w_offset', 'loss_constant', and 'R'. If None, default values will
        be used.
    bounds :: dict
        A dictionary containing the bounds for the model parameters during
        fitting. The expected keys are the same as for p0, and the values should
        be tuples specifying the (min, max) bounds for each parameter. If None,
        default bounds will be used.
    max_iterations :: int
        The maximum number of iterations for sigma-clipping routines to
        remove outliers when fitting the extinction model and achromatic
        deviations. Should be a positive integer.
    outlier_threshold :: float
        The number of MADs above which a point is considered an outlier
        when fitting the extinction model and achromatic deviations. Should be a
        positive float.
    return_components :: bool
        If True, also returns the individual extinction components (O2, H2O,
        Rayleigh) as separate arrays in a dictionary. Default is False.
    debug :: bool
        Enables optional debugging plots for the extinction model fitting and
        achromatic deviation fitting processes.

    Returns:
    --------
    atmospheric_correction_model :: np.ndarray
        A 2D array with the same shape as 'flux' and 'error' that contains the
        multiplicative correction factor to remove atmospheric losses.
    extinction_components :: dict
        A dictionary containing the individual extinction components as separate
        arrays with keys 'o2', 'humidity', and 'rayleigh'. Only returned if
        'return_components' is True.
    achromatic_correction :: np.ndarray
        A 2D array with the same shape as 'flux' and 'error' that contains the
        multiplicative correction factor to remove achromatic bumps. Only
        returned if 'return_components' is True.
    """

    # Step 1: Estimate extinction coefficients (taus) for each wavelength bin
    taus, taus_error = estimate_extinction_coefficients(
        airmass=airmass,
        flux=flux,
        error=error,
        max_iterations=max_iterations,
        clip_threshold=outlier_threshold,
    )

    # Step 2: Fit the extinction model to the estimated taus
    throughput_model, fitted_values, _ = fit_extinction_model(
        wavelengths=wavelengths,
        taus=taus,
        taus_error=taus_error,
        p0=p0,
        bounds=bounds,
        debug=debug,
    )

    # Step 3: Calculate average telluric loss at each airmass
    chromatic_correction = np.array([throughput_model(X) for X in airmass])

    # Step 4: Fit achromatic bumps to the residuals after removing chromatic effects
    achromatic_correction, _ = fit_achromatic_bumps(
        times,
        flux,
        error,
        airmass,
        throughput_model,
        max_iterations=max_iterations,
        outlier_threshold=outlier_threshold,
        debug=debug,
    )

    # Step 5: Combine chromatic and achromatic corrections
    atmospheric_correction_model = achromatic_correction / chromatic_correction

    # Optional: Return individual extinction components if requested
    if return_components:

        # These are generated using the best-fit extinction parameters
        o2_component = generate_extinction_model(
            wavelengths=wavelengths,
            rs_tau0=0.0,
            o2_abundance=fitted_values[1],
            humidity=0.0,
            w_offset=0.0,
            loss_constant=1.0,
            R=fitted_values[5],
        )
        humidity_component = generate_extinction_model(
            wavelengths=wavelengths,
            rs_tau0=0.0,
            o2_abundance=0.0,
            humidity=fitted_values[2],
            w_offset=0.0,
            loss_constant=1.0,
            R=fitted_values[5],
        )
        rayleigh_component = generate_extinction_model(
            wavelengths=wavelengths,
            rs_tau0=fitted_values[0],
            o2_abundance=0.0,
            humidity=0.0,
            w_offset=0.0,
            loss_constant=fitted_values[4],
            R=fitted_values[5],
        )

        # This seems like the easiest way to keep track of different components
        extinction_components = {
            "o2": o2_component,
            "humidity": humidity_component,
            "rayleigh": rayleigh_component,
        }

        return (
            atmospheric_correction_model,
            extinction_components,
            achromatic_correction,
        )

    return atmospheric_correction_model
