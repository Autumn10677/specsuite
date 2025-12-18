import matplotlib.pyplot as plt
import numpy as np
import warnings


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
