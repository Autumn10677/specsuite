import matplotlib.pyplot as plt
import numpy as np


def plot_image(
    image,
    xlim=None,
    ylim=None,
    xlabel=None,
    ylabel=None,
    cbar_label="Counts",
    title="",
    figsize=(10, 3),
    cmap=None,
    **kwargs,
):

    if cmap is None:
        cmap = "inferno"
    if xlim is None:
        xlim = [0, len(image[0])]
    if ylim is None:
        ylim = [0, len(image)]
    if xlabel is None:
        xlabel = "Dispersion Axis (pix)"
    if ylabel is None:
        ylabel = "Cross-Dispersion Axis (pix)"

    plt.rcParams["figure.figsize"] = figsize
    plt.imshow(
        image,
        cmap="inferno",
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
    plt.show()


def plot_spectra(flux, err, p_wavecal=None, plot_idx=0):

    # Adjusts the xlabel / x-data if necessary
    xlabel = "Dispersion Axis (pix)"
    xs = np.array(range(len(flux)))
    if p_wavecal is not None:
        xs = p_wavecal(xs)
        xlabel = "Wavelength (AA)"

    # Plots spectra with errorbars
    plt.rcParams["figure.figsize"] = (12, 5)
    plt.scatter(xs, flux.T[plot_idx], color="k", s=3)
    plt.errorbar(xs, flux.T[plot_idx], yerr=err.T[plot_idx], fmt="none", color="k")
    plt.xlim(xs[0], xs[-1])
    plt.xlabel(xlabel)
    plt.show()


def _gaussian(x, A, mu, sigma):
    "Generates a 1D Gaussian curve at each point in x"
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def _moffat(x, A, mu, gamma):
    "Generates a 1D modified Moffat curve at each point in x"
    return A * (1 + ((x - mu) / gamma) ** 2) ** (-2.5)


def rebin_image_columns(image, bin):
    """
    Rebins an image along a single axis. The bin
    size must be an integer multiple of the axis
    size being rebinned.

    Parameters:
    -----------
    image :: np.ndarray
        Original image to be rebinned.
    bin :: int
        Size each bin in pixels along the
        columns of the provided image.

    Returns:
    --------
    rebinned_image :: np.ndarray
        An image where the columns have been
        rebinned into bin length pixels.
    """

    # Initializes list for rebinned columns
    rebinned_columns = []

    # Loop over the columns (for each bin)
    for i in range(int(len(image[0]) / bin)):
        subim = np.median(image[:, i * bin : (i + 1) * bin], axis=1)
        rebinned_columns.append(subim)

    # Stacks all columns into one rebinned image
    rebinned_image = np.column_stack(rebinned_columns)

    return rebinned_image


def flatfield_correction(image, flat, debug=False):
    """
    Parameters:
    -----------
    image :: np.ndarray
        Image(s) that should be divided by the
        normalized flatfield image. This can be
        a single 2D image or an array of 2D images.
    flat :: np.ndarray
        A single unnormalized flatfield image,
        ideally the median of several flatfield
        exposures.]
    debug :: bool
        Allows for diagnostic plotting.

    Returns:
    --------
    flatfielded_ims :: np.ndarray
        The resulting image(s) after being divided
        by the normalized flatfield
    """

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
