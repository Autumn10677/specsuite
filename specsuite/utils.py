from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import subprocess
import warnings
import os


def plot_image(
    image: np.ndarray,
    xlim: tuple = None,
    ylim: tuple = None,
    xlabel: str = "Dispersion Axis (pix)",
    ylabel: str = "Cross-Dispersion Axis (pix)",
    cbar_label: str = "Counts",
    title: str = "",
    figsize: tuple = (10, 3),
    cmap: str = "inferno",
    savedir: str = None,
    **kwargs,
):
    """
    A simple wrapper for matplotlib.pyplot.imshow(). By default, this
    function uses a handful of style options to keep all visualizations
    consistent within our documentation. You should be able to
    overwrite these options and provide any of the standard additional
    KWARGS.

    Parameters:
    -----------
    image :: np.ndarray
        A single 2D array. If it is not a Numpy array, the function
        will attempt to convert it into one.
    xlim :: tuple
        The (xmin, xmax) to show. If none is provided, defaults to the
        entire horizontal span of the image.
    ylim :: tuple
        The (ymin, ymax) to show. If none is provided, defaults to the
        entire vertical span of the image.
    xlabel :: str
        Text to write along the x-axis (bottom) of the image.
    ylabel :: str
        Text to write along the y-axis (left) of the image.
    cbar_label :: str
        A text label assigned to the colorbar.
    title :: str
        A title to plot at the top of the image.
    figsize :: tuple
        The dimensions (horizontal, vertical) of the image.
    cmap :: str
        Name of the matplotlib colormap to use.
    savedir :: str
        Directory (+filename) to save the generated image at. If an argument
        is provided, then 'plt.show()' will not run.
    """

    try:

        image = np.array(image).astype(float)
        assert len(image.shape) == 2

        # Necessary to prevent weird behavior at edges of image
        if xlim is None:
            xlim = [-0.5, len(image[0]) - 0.5]
        if ylim is None:
            ylim = [-0.5, len(image) - 0.5]

        plt.rcParams["figure.figsize"] = figsize
        plt.imshow(
            image,
            cmap=cmap,
            aspect="auto",
            interpolation="none",
            origin="lower",
            **kwargs,
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar(label=cbar_label)
        plt.xlim(xlim)
        plt.ylim(ylim)

        if savedir is not None:
            plt.savefig(savedir, bbox_inches="tight")
            plt.clf()
            plt.close()
        else:
            plt.show()

    except AssertionError:
        warnings.warn("The provided image is not a valid 2D array")


def animate_images(
    image_array: np.ndarray,
    delay: int = 10,
    savedir: str = "result.gif",
    iterable: list = None,
    iterable_label: str = "Index",
    **kwargs,
):
    """
    Attempts to create a GIF from an array of 2D Numpy arrays.
    This creates a series of temporary images in a '__TEMP_FRAMES__'
    directory, then uses 'magick' to convert them to a GIF. These
    frames should be deleted automatically once the GIF has been
    created. All '**kwargs' will be fed directly into
    'specsuite.plot_image'.

    Parameters:
    -----------
    image_array :: np.ndarray
        An array containing several 2D images. Each individual image
        will become a frame in the resulting GIF. Additionally, the
        GIF will preserve the order of this array.
    delay :: int
        The time (1/100 seconds) between frames.
    savedir :: str
        The directory (+filename) to save the GIF under.
    iterable :: list
        A 1D list with the same length as 'image_array'. This will be
        plotted at the top of the plot and updates with every frame.
        If none is provided, this defaults to the image index.
    iterable_label :: str
        The 'name' you would like to associate with the 'iterable'.
        Defaults to 'Index'.
    """

    # Defaults to frame number
    if iterable is None:
        iterable = np.arange(1, len(image_array) + 1)

    # If not required, function could fail without deleting temporary files
    assert len(image_array) == len(
        iterable
    ), "Image array and iterable must have the same length"

    # This should only trigger if function failed to finish before
    try:
        os.mkdir("__TEMP_FRAMES__")
    except FileExistsError:
        pass

    # Saves every file in a temporary folder
    for idx, im in enumerate(image_array):
        plot_image(
            im,
            title=f"{iterable_label}: {iterable[idx]:04d}",
            savedir=f"__TEMP_FRAMES__/{idx:04d}.png",
            **kwargs,
        )

    # This runs a terminal command (creates the GIF using 'magick')
    subprocess.run(["magick", "-delay", str(delay), "__TEMP_FRAMES__/*.png", savedir])

    # Removes the temporary files / directory
    for idx, _ in enumerate(image_array):
        os.remove(f"__TEMP_FRAMES__/{idx:04d}.png")
    os.rmdir("__TEMP_FRAMES__")


def _gaussian(x: np.ndarray, A: float, mu: float, sigma: float) -> np.ndarray:
    """
    Generates a 1D Gaussian profile on the user-provided grid of
    x-points. If an error is encountered, then 'None' will be returned
    instead of a Numpy array.

    Parameters:
    -----------
    x :: np.ndarray
        A set of x-points over which to evaluate the Gaussian profile.
        This can be a single value, but must still be contained in a
        list (i.e., [1]).
    A :: float
        The amplitude of the Gaussian profile.
    mu :: float
        The mean of the Gaussian profile.
    sigma :: float
        The standard deviation of the Gaussian profile.

    Returns:
    --------
    profile :: np.ndarray
        The 1D Gaussian profile evaluated on the provided grid of x-points.
    """

    # Ensures the calculation can run without error
    try:
        x = np.array(x).astype(float)
        A, mu, sigma = np.array([A, mu, sigma]).astype(float)
    except ValueError:
        return None

    profile = A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    return profile


def _moffat(
    x: np.ndarray,
    A: float,
    mu: float,
    gamma: float,
    offset: float = 0.0,
) -> np.ndarray:
    """
    Generates a 1D Moffat profile on the user-provided grid of
    x-points. If an error is encountered, then 'None' will be returned
    instead of a Numpy array. Note: This is technically a 'modified
    Moffat profile' since the exponent has been set to 2.5.

    Parameters:
    -----------
    x :: np.ndarray
        A set of x-points over which to evaluate the Moffat profile.
        This can be a single value, but must still be contained in a
        list (i.e., [1]).
    A :: float
        The amplitude of the Moffat profile.
    mu :: float
        The mean of the Moffat profile.
    gamma :: float
        A shape parameter for the Moffat profile.
    offset :: float
        A constant offset applied to all points.

    Returns:
    --------
    profile :: np.ndarray
        The 1D Moffat profile evaluated on the provided grid of x-points.
    """

    # Ensures the calculation can run without error
    try:
        x = np.array(x).astype(float)
        A, mu, gamma = np.array([A, mu, gamma]).astype(float)
    except ValueError:
        return None

    profile = A * (1 + ((x - mu) / gamma) ** 2) ** (-2.5) + offset

    return profile


def rebin_image_columns(image: np.ndarray, bin: int) -> np.ndarray:
    """
    Rebins an image along a single axis. The bin size must be an
    integer multiple of the axis size being rebinned.

    Parameters:
    -----------
    image :: np.ndarray
        Original image to be rebinned.
    bin :: int
        Size each bin in pixels along the columns of the provided
        image.

    Returns:
    --------
    rebinned_image :: np.ndarray
        An image where the columns have been rebinned into bin length
        pixels.
    """

    assert isinstance(bin, int), f"Bin size must be an int, not {type(bin)}"

    # Initializes list for rebinned columns
    rebinned_columns = []

    # Loop over the columns (for each bin)
    for i in range(int(len(image[0]) / bin)):
        subim = np.median(image[:, i * bin : (i + 1) * bin], axis=1)
        rebinned_columns.append(subim)

    # Stacks all columns into one rebinned image
    rebinned_image = np.column_stack(rebinned_columns)

    return rebinned_image


def flatfield_correction(
    image: np.ndarray, flat: np.ndarray, debug: bool = False
) -> np.ndarray:
    """
    Applies a simple flatfield correction to one or more 2D images.
    This function assumes that each entry along the first axis is a 2D
    image with the same size as 'flat'.

    Parameters:
    -----------
    image :: np.ndarray
        Image(s) that should be divided by the normalized flatfield
        image. This can be a single 2D image or an array of 2D images.
    flat :: np.ndarray
        A single unnormalized flatfield image, ideally the median of
        several flatfield exposures.
    debug :: bool
        Allows for diagnostic plotting.

    Returns:
    --------
    flatfielded_ims :: np.ndarray
        The resulting image(s) after being divided by the normalized
        flatfield.
    """

    image = np.array(image)
    flat = np.array(flat)

    assert image.shape[-2:] == flat.shape, (
        "Image(s) and flatfield are not compatible shapes"
        f"({image.shape} vs. {flat.shape})"
    )

    # Calculates flatfield corrections
    normed_flat = flat / np.median(flat, axis=0)
    flatfielded_ims = image / normed_flat

    # Plots diagnostic images
    if debug:

        # Calculates statistics used for colorbars
        median_flux = np.median(normed_flat)
        std_flux = np.std(normed_flat)
        plot_image(
            normed_flat,
            title="Normalized Flatfield",
            vmin=median_flux - 4 * std_flux,
            vmax=median_flux + 4 * std_flux,
        )

    return flatfielded_ims


def peak_phase_shift(
    ref_fft: np.ndarray,
    data_fft: np.ndarray,
    alpha: float = 0.5,
) -> float:
    """
    Estimates the shift between two signals by finding the peak of the
    phase correlation. This is a coarse estimate that can be refined
    using the 'phase_slope_shift' function.

    Parameters:
    -----------
    ref_fft :: np.ndarray
        The Fourier transform of the reference signal.
    data_fft :: np.ndarray
        The Fourier transform of the data signal.
    alpha :: float
        A power to apply to the magnitude of the cross-correlation. This

    Returns:
    --------
    shift :: float
        The estimated shift between the two signals. This can be a
        non-integer value due to sub-pixel refinement.
    """

    # Calculates the phase correlation and applies a power to the magnitude
    cross = np.conjugate(ref_fft) * data_fft
    R = cross / (np.abs(cross) ** alpha + 1e-12)
    corr = np.real(np.fft.ifft(R))

    # Finds the index of the peak correlation, adjusts if necessary
    i = np.argmax(corr)
    N = len(corr)
    if i > N // 2:
        i -= N

    # Sub-pixel quadratic refinement
    im1 = (i - 1) % N
    ip1 = (i + 1) % N
    y0, y1, y2 = corr[im1], corr[i], corr[ip1]

    # If the parabola is too flat, then the sub-pixel shift is likely not meaningful
    denom = y0 - 2 * y1 + y2
    if np.abs(denom) < 1e-12:
        return float(i)

    # Calculates the sub-pixel shift by finding the vertex of the parabola
    sub = 0.5 * (y0 - y2) / denom
    shift = i + sub

    return shift


def phase_slope_shift(
    ref_fft: np.ndarray,
    data_fft: np.ndarray,
) -> float:
    """
    Estimates the sub-pixel shift between two signals by calculating the
    slope of the phase correlation. This is a refinement step that can be
    applied after finding the peak shift using 'peak_phase_shift'.

    Parameters:
    -----------
    ref_fft :: np.ndarray
        The Fourier transform of the reference signal.
    data_fft :: np.ndarray
        The Fourier transform of the data signal.

    Returns:
    --------
    shift :: float
        An estimate of the sub-pixel shift between the two signals.
    """

    # Calculates the phase correlation and unwraps it to prevent discontinuities
    ratio = data_fft / ref_fft
    phase = np.unwrap(np.angle(ratio))
    freq = np.fft.fftfreq(len(phase))

    # Finds frequencies within a reasonable bandpass to mitigate noise/systematics
    power = np.abs(ref_fft) ** 2
    low = np.percentile(power, 10)
    high = np.percentile(power, 95)

    # Masks undesired badpass
    mask = (power > low) & (power < high)
    mask &= freq != 0

    # Triggers if too few frequencies are left over after masking
    if np.sum(mask) < 5:
        return 0.0

    slope, _ = np.polyfit(freq[mask], phase[mask], 1)
    return -slope / (2 * np.pi)


def estimate_shift(
    ref_fft: np.ndarray,
    data_fft: np.ndarray,
) -> float:
    """
    Attempts to estimate the sub-pixel shift between two signals.
    This is done in two steps, first by finding the peak coarse
    shift, then by refining this estimate using the slope of the phase
    correlation.

    Parameters:
    -----------
    ref_fft :: np.ndarray
        The Fourier transform of the reference signal.
    data_fft :: np.ndarray
        The Fourier transform of the data signal.

    Returns:
    --------
    shift :: float
        The estimated sub-pixel shift between the two signals.
    """

    # Coarse, global estimate
    coarse = peak_phase_shift(ref_fft, data_fft)

    # Recenter data FFT
    freq = np.fft.fftfreq(len(ref_fft))
    data_fft_centered = data_fft * np.exp(2j * np.pi * freq * coarse)

    # Fine, local estimate
    fine = phase_slope_shift(ref_fft, data_fft_centered)

    # Enforce validity
    if not np.isfinite(fine) or abs(fine) > 0.5:
        fine = 0.0

    return coarse + fine


def convolve_to_resolution(
    x: np.ndarray,
    y: np.ndarray,
    R: float,
):
    """
    Convolves an input spectrum using a Gaussian kernel where the
    kernel width is determined by the desired resolution R.

    Parameters:
    -----------
    x :: np.ndarray
        Wavelength array (in Angstroms).
    y :: np.ndarray
        Flux array corresponding to the wavelengths.
    R :: float
        Desired spectral resolution (lambda/delta_lambda).

    Returns:
    --------
    y_conv :: np.ndarray
        Convolved flux array at the same wavelengths.
    """

    # These are logical assertions, not technically required to run, though
    assert len(x) == len(y), "Wavelength and flux arrays must have the same length!"
    assert R > 0, "Resolution R must be a positive number!"

    # Checks for NaN values, which would cause the convolution to fail
    assert not np.any(np.isnan(x)), "Wavelength array cannot contain NaN values!"
    assert not np.any(np.isnan(y)), "Flux array cannot contain NaN values!"

    # Compute the log-lambda spacing
    loglam = np.log(x)
    dloglam = np.mean(np.diff(loglam))

    # Convert R to sigma in log-lambda space
    sigma_loglam = 1 / R
    sigma_pixels = sigma_loglam / dloglam

    # Match scipy default: truncate at 4 sigma
    truncate = 4.0
    radius = int(truncate * sigma_pixels + 0.5)

    # Build Gaussian kernel
    grid = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (grid / sigma_pixels) ** 2)
    kernel /= np.sum(kernel)

    # Reflect padding to mimic gaussian_filter1d(mode='reflect')
    y_padded = np.pad(y, pad_width=radius, mode="reflect")

    # FFT convolution
    y_conv = fftconvolve(y_padded, kernel, mode="same")[radius:-radius]

    return y_conv


def correct_lightcurve_shifting(
    flux: np.ndarray,
    error: np.ndarray,
    N_divisions: int = 1,
    model_flux: np.ndarray = None,
    mode: str = "median",
    poly_order: int = 3,
    apply_window: bool = True,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Attempts to model the sub-pixel offsets between exposures in
    a spectrophotometric observation. Flux will be conserved during
    interpolation, meaning that non-constant offsets can create
    sudden jumps/drops in flux for pixels that are larger/smaller
    than they initially were. For any 'invalid' values that would
    require interpolating outside the range of the provided flux
    array will be returned as a NaN.

    Parameters:
    -----------
    flux :: np.ndarray
        A 2D array of fluxes oriented so that the dimensions
        represent (N_exposures, N_pixels).
    error :: np.ndarray
        A 2D array of errors for each entry in the 'flux' array.
    N_divisions :: int
        The number of equal chunks to split each exposure into
        to infer the offset. Generally, large values (N > 5)
        fail to accurately infer sub-pixel offsets due to a lack
        of distinct features in each sub-region.
    model_flux :: np.ndarray
        A model for the flux present in each exposure. If 'None',
        this defaults to the average flux at each pixel across
        every exposure.
    mode :: str
        Determines how sub-pixel offset information will be
        handled. 'median' calculates the median sub-pixel offset
        across each sub-region. 'fit' will use a polynomial with
        an order given by 'poly_order' to infer how offset varies
        along the dispersion axis of your flux array.
    poly_order :: int
        The polynomial order to use for 'fit' mode. To prevent
        issues with over-fitting, 'poly_order' must be less than
        or equal to 'N_divisions'.
    apply_window :: bool
        Optionally disables the 'Hanning' window that is applied
        to sub-regions before performing phase correlation.
    progress :: bool
        Toggles the progress bar.

    Returns:
    --------
    interpolated_flux :: np.ndarray
        Interpolated flux at each valid 'target_pixels' location.
    interpolated_error :: np.ndarray
        Propagated 1-sigma uncertainties of 'interp_flux'.
    """

    # Finds a 2D array of 'effective pixels' for each exposure
    current_pixels = estimate_exposure_offsets(
        flux=flux,
        model_flux=model_flux,
        N_divisions=N_divisions,
        mode=mode,
        poly_order=poly_order,
        apply_window=apply_window,
        progress=progress,
    )

    # Interpolates all exposures onto a single, shared pixel grid
    interpolated_flux, interpolated_error = apply_new_pixel_grid(
        flux=flux,
        error=error,
        current_pixels=current_pixels,
        progress=progress,
    )

    return interpolated_flux, interpolated_error


def estimate_exposure_offsets(
    flux: np.ndarray,
    model_flux: np.ndarray = None,
    N_divisions: int = 1,
    mode: str = "median",
    poly_order: int = 3,
    apply_window: bool = True,
    progress: bool = False,
) -> np.ndarray:
    """
    Attempts to estimate the effective pixel positions of the
    input 'flux' array compared to the 'model_flux'. This is
    done by performing a phase correlation of a model spectra
    against each individual exposure. The phase correlation
    produces an estimate of the sub-pixel offset(s) between
    the model and a given exposure. If 'N_divisions' > 1,
    this process is performed multiple sub-regions in every
    single exposure.

    Parameters:
    -----------
    flux :: np.ndarray
        A 2D array of fluxes oriented so that the dimensions
        represent (N_exposures, N_pixels).
    model_flux :: np.ndarray
        A 1D flux model to use for estimating sub-pixel offsets.
        If no model is provided, the median across all exposures
        will be used by default.
    N_divisions :: int
        The number of equal segments to split each individual
        exposure into before estimating pixel offsets.
    mode :: str
        Controls how sub-pixel offset information is used. If
        'median', the median sub-pixel offset across all sub-regions
        is used for a given exposure. If 'fit', a polynomial is fit
        to extracted offsets and used to infer how offsets change
        the 'pixel axis' of the data.
    poly_order :: int
        The order of polynomial to fit to sub-pixel offets. Only
        matters if 'fit' mode is used, and must be <= 'N_divisions'
        to prevent over-fitting errors.
    apply_window :: bool
        Controls whether or not the 'hanning' window is used to smooth
        out edge effects before performing phase correlation. Defaults
        to 'True'.
    progress :: bool
        Toggles the progress bar.

    Returns:
    --------
    offset_pixels :: np.ndarray
        The 'true' pixel array that each exposure in 'flux' lies on.
        These are defined relative to the 'model_flux', and may contain
        values outside of the model's pixel locations.
    """

    # Prevents users from running the calculation before failing
    assert mode in [
        "median",
        "fit",
    ], f"Mode '{mode}' not supported, must be 'median' or 'fit'!"
    if not (model_flux is None):
        assert (
            flux.shape[1] == model_flux.shape[0]
        ), "'model_flux' must have the same shape as every individual 'flux' entry!"

    # These checks only matter if performing a polynomial fit
    if mode == "fit":
        assert isinstance(poly_order, int), "'poly_order' must be an integer!"
        assert (
            poly_order <= N_divisions
        ), "'poly_order' must be <= 'N_divisions' to prevent overfitting!"

    # Makes later code easier to read
    N_exposures, N_pixels = flux.shape

    # Initialized here to allow in-place operations
    bin_centers = np.full(N_divisions, np.nan)
    offsets_array = np.zeros((N_exposures, N_divisions))

    # Initializes pixel arrays
    pixels = np.arange(N_pixels)
    offset_pixels = np.array([pixels for _ in range(N_exposures)])

    # This is can be an issue if many NaNs exist in 'flux'
    if model_flux is None:
        model_flux = np.nanmedian(flux, axis=0)

    # Determines the size of each segment for splitting the SEDs
    pad = len(pixels) // N_divisions

    # Splits SEDs into N_divisions segments for analysis
    for idx in tqdm(
        range(N_divisions),
        desc="Inferring sub-pixel offsets",
        disable=(not progress),
    ):

        # Extracts portion of 'flux_model' needed for this sub-region
        start, end = (idx * pad, (idx + 1) * pad)
        bin_centers[idx] = (start + end) / 2
        flux_reference = model_flux[start:end].copy()

        # The 'hanning' window is used to mitigate FFT edge effects
        if apply_window:
            window = np.hanning(len(flux_reference))
            flux_reference *= window
        reference_freqs = np.fft.fft(flux_reference)

        offsets_temp = []
        for row in flux:

            # Applies 'hanning' window to individual sub-region
            flux_data = row.copy()[start:end].copy()
            if apply_window:
                flux_data *= window

            # Estimates sub-pixel offsets
            data_fft = np.fft.fft(flux_data)
            shift = estimate_shift(reference_freqs, data_fft)
            offsets_temp.append(-shift)

        offsets_array[:, idx] = offsets_temp

    # A constant offset is assigned to each exposure's pixels grid
    if mode == "median":
        median_offsets = np.median(offsets_array, axis=1)
        offset_pixels = np.array(
            [pix + o for pix, o in zip(offset_pixels, median_offsets)]
        )

    # Infers pixel offsets continuously using a polynomial fit
    elif mode == "fit":

        # Initializing with NaNs to intentionally break if 'np.polyfit' fails
        coeffs = np.full((N_exposures, poly_order + 1), np.nan)

        # Should only fail if 'np.polyfit()' does not converge
        for idx in range(N_exposures):
            coeffs[idx] = np.polyfit(bin_centers, offsets_array[idx], poly_order)
            offset_pixels[idx] = offset_pixels[idx] + np.poly1d(coeffs[idx])(pixels)

    return offset_pixels


def perform_cdf_interpolation(
    flux: np.ndarray,
    error: np.ndarray,
    current_pixels: np.ndarray,
    target_pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flux-conserving interpolation using linear interpolation
    of the cumulative distribution function (CDF). This function
    assumes that x-errors are negligible and that the input
    flux array has errors that are independent and random. Any
    'target_pixels' that fall outside the range spanned by
    'current_pixels' will be returned with a NaN value.

    Parameters:
    -----------
    flux :: np.ndarray
        The original flux values.
    error :: np.ndarray
        1-sigma uncertainties corresponding to `flux`.
    current_pixels :: np.ndarray
        The pixels locations of that the 'flux' and 'error' arrays
        are currently sampled on.
    target_pixels :: np.ndarray
        The desired pixels to interpolate onto. Only locations (xd)
        that lie within the range spanned by 'current_pixels' will
        be calculated.

    Returns:
    --------
    interp_flux :: np.ndarray
        Interpolated flux at each valid 'target_pixels' location.
    interp_error :: np.ndarray
        Propagated 1-sigma uncertainties of 'interp_flux'.
    """

    # Ensures that all inputs are Numpy arrays of floats
    flux = np.asarray(flux, dtype=float)
    error = np.asarray(error, dtype=float)
    x = np.asarray(current_pixels, dtype=float)
    xd = np.asarray(target_pixels, dtype=float)

    # Only checks flux since shape checks will trigger other mismatches
    assert flux.ndim == 1, "All arrays must be 1-dimensional!"

    # If lengths are different, interpolation will throw index errors
    assert (
        current_pixels.shape == flux.shape
    ), "There is a shape mismatch between 'current_pixels' and 'flux'!"
    assert (
        flux.shape == error.shape
    ), "'flux' and 'error' arrays should be the same shape!"

    # Ensures that 'np.searchsorted()' will work as intended
    assert np.all(x[:-1] <= x[1:]), "'current_pixels' is not sorted!"
    assert np.all(xd[:-1] <= xd[1:]), "'target_pixels' is not sorted!"

    # Uses fancy approximation, see their documentation!
    dx = np.gradient(x)
    dxd = np.gradient(xd)

    # Converts flux into a non-normalizedCDF
    cdf = np.cumsum(flux * dx)
    cdf_var = np.cumsum((error * dx) ** 2)

    # Requires sorted x-arrays, should 'mask' invalid x-values
    idx = np.searchsorted(x, xd)
    valid = (idx > 0) & (idx < len(x))

    # Stores these in arrays for quick access later
    idxL = idx[valid] - 1
    idxR = idx[valid]
    xL = x[idxL]
    xR = x[idxR]

    # Similar thing here, vectorized calculation helps later
    wL = (xR - xd[valid]) / (xR - xL)
    wR = (xd[valid] - xL) / (xR - xL)

    # 'Invalid' values will have a NaN value
    interp_cdf = np.full(len(xd), np.nan)
    interp_cdf[valid] = wL * cdf[idxL] + wR * cdf[idxR]

    # 'Invalid' values will also have NaN variances
    interp_cdf_var = np.full(len(xd), np.nan)
    interp_cdf_var[valid] = (
        wL**2 * cdf_var[idxL] + wR**2 * cdf_var[idxR] + 2 * wL * wR * cdf_var[idxL]
    )

    # Dividing by 'dxd' should handle non-linear offsets
    interp_flux = np.full(len(xd), np.nan)
    interp_flux[1:] = (interp_cdf[1:] - interp_cdf[:-1]) / dxd[1:]
    interp_flux[0] = interp_cdf[0]

    # Initializes final variance array + finds indices to iterate over
    interp_var = np.full(len(xd), np.nan)
    valid_idx = np.where(valid)[0]

    # Iterates over every 'valid' interpolated value
    for k in range(1, len(valid_idx)):

        # Used to find relevant interpolated CDF variances
        i_curr = valid_idx[k]
        i_prev = valid_idx[k - 1]

        # Current interpolation weights/indices
        aL, aR = idxL[k], idxR[k]
        waL, waR = wL[k], wR[k]

        # Previous interpolation weights/indices
        bL, bR = idxL[k - 1], idxR[k - 1]
        wbL, wbR = wL[k - 1], wR[k - 1]

        # Covariance between adjacent interpolated CDF values
        cov = (
            waL * wbL * cdf_var[min(aL, bL)]
            + waL * wbR * cdf_var[min(aL, bR)]
            + waR * wbL * cdf_var[min(aR, bL)]
            + waR * wbR * cdf_var[min(aR, bR)]
        )

        # Final variance of interpolated fluxes
        interp_var[i_curr] = (
            interp_cdf_var[i_curr] + interp_cdf_var[i_prev] - 2 * cov
        ) / (dxd[i_curr] ** 2)

    # Edge handling
    interp_var[0] = interp_var[1]

    # Converts variance into standard deviations
    interp_error = np.sqrt(interp_var)

    return interp_flux, interp_error


def apply_new_pixel_grid(
    flux: np.ndarray,
    error: np.ndarray,
    current_pixels: np.ndarray,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Corrects for pixel offsets between individual exposures
    by interpolating every exposure onto the same sub-pixel
    grid. This is done by linearly interpolating on the
    unnormalized cummulative distribution function of each
    exposure.

    Parameters:
    -----------
    flux :: np.ndarray
        A 2D array of fluxes oriented so that the dimensions
        represent (N_exposures, N_pixels).
    error :: np.ndarray
        A 2D array of 1-sigma uncertainties with the same
        dimensions as 'flux'.
    current_pixels :: np.ndarray
        A 2D array containing the 'effective pixel positions'
        of the 'flux' and 'error' arrays.
    progress :: bool
        Toggles the progress bar.

    Returns:
    --------
    interpolated_flux :: np.ndarray
        A 2D array of fluxes interpolated onto a single pixel
        grid. May contain NaN values if any 'current_pixels'
        cannot be linearly interpolated without extrapolating
        outside the assumed pixel bounds.
    interpolated_error :: np.ndarray
        A 2D array of error-propagated errors for the
        'interpolated_flux' array.
    """

    assert flux.ndim == 2, "'flux' array must be 2D!"
    assert (
        flux.shape == error.shape
    ), "'flux' and 'error' arrays should be the same shape!"

    # Makes later code easier to read
    N_exposures, N_pixels = flux.shape

    # Assumes 'desired pixels' are just [0, 1, 2, ...]
    target_pixels = np.array(range(N_pixels))

    # Allows for in-place assignment
    interpolated_flux = np.zeros((N_exposures, N_pixels))
    interpolated_error = np.zeros((N_exposures, N_pixels))

    # Interpolates each exposure onto new pixel grid
    for idx in tqdm(
        range(N_exposures),
        desc="Interpolating onto new grid",
        disable=(not progress),
    ):
        f_temp, e_temp = perform_cdf_interpolation(
            flux=flux[idx],
            error=error[idx],
            current_pixels=current_pixels[idx],
            target_pixels=target_pixels,
        )
        interpolated_flux[idx] = f_temp
        interpolated_error[idx] = e_temp

    return interpolated_flux, interpolated_error
