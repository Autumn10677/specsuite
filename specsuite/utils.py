from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
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
