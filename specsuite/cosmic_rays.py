import matplotlib.patches as patches
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

from collections import defaultdict
from astropy.stats import mad_std
from tqdm import tqdm
from jax import lax


def _gaussian_mask(
    shape: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Create an N-dimensional Gaussian mask where the mean
    and standard deviation is specified along each
    dimension.

    Parameters:
    -----------
    shape :: np.ndarray
        An array of N integers that specifies the shape
        of the mask.
    mean :: np.ndarray
        An array of N floats that control the mean of the
        gaussian profiles along each dimension. Must have
        the same number of entries as 'shape'.
    std :: np.ndarray
        An array of N floats indicating the standard
        deviation of the gaussian profiles along each
        dimension.

    Returns:
    --------
    gaussian :: np.ndarray
        An n-dimensional NumPy array with Gaussian values
    """

    # Prevents errors due to inhomogeneous dimensions
    assert len(shape) == len(mean) == len(std), "All tuples must be the same length!"

    # Creates an n-dimensional mesh grid for the Gaussian mask
    coords = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")

    # Determines the gaussian exponent for all points
    squared_diffs = [
        (coord - mu) ** 2 / (2 * sigma**2)
        for coord, mu, sigma in zip(coords, mean, std)
    ]

    # Calculates and normalizes the n-dimensional gaussian
    exponent = sum(squared_diffs)
    gaussian = np.exp(-exponent)
    gaussian = (gaussian - np.min(gaussian)) / (np.max(gaussian) - np.min(gaussian))

    return gaussian


def _expand_mask(mask: np.ndarray, ray_padding: int = 1) -> np.ndarray:
    """
    Uses JAX convolution to pad zero-valued pixels
    surrounding nonzero-valued pixels. This can
    be performed multiple times to produce several
    layers of padding.

    Parameters:
    -----------
    mask :: np.ndarray
        An array with binary (1 or 0) entries.
    ray_padding :: int
        How many pixels surrounding a cosmic ray
        to flag as 'potentially contaminated.'
        These padded pixels will all have a value
        between 0 and 1 to allow users to easily
        filter them out after processing.

    Returns:
    --------
    mask :: np.ndarray
        A nearly-identical array to the input mask,
        but with non-zero values for pixels surrounding
        each non-zero entry in the original array.
    """

    # Defines a kernel used to check all neighboring pixels
    kernel = jnp.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=jnp.float32).reshape(
        (1, 1, 3, 3)
    )

    # Prepares mask for convolution
    mask = mask.astype(jnp.float32)[None, None, :, :]

    # Pads the mask data for the specified number of iterations
    padded_value = 0.5
    for _ in range(ray_padding):

        # Finds and updates pixels w/ a value of 0 and nonzero neighbors
        neighbors = lax.conv_general_dilated(mask, kernel, (1, 1), "SAME")
        mask = jnp.where((mask == 0) & (neighbors > 0), padded_value, mask)
        padded_value /= 2

    return mask[0, 0]


def _cluster_in_grid(
    x_points: np.ndarray,
    y_points: np.ndarray,
    R: float,
) -> np.ndarray:
    """
    Groups an array of points that lie within a box of
    width R around one another.

    Parameters:
    -----------
    x_points :: np.ndarray
        All the x-coordinates of points to be
        grouped.
    y_points :: np.ndarray
        All the y-coordinates of points to be
        grouped.
    R :: float
        Width of the box used for grouping points
        together.

    Returns:
    --------
    averaged_points :: np.ndarray()
        An array of averaged points.
    """

    # Defines data structures for storing grouped points
    points = np.column_stack((x_points, y_points))
    grid = defaultdict(list)

    # Assigns points to grid cells
    for point in points:
        gx, gy = int(point[0] // R), int(point[1] // R)
        grid[(gx, gy)].append(point)

    # Averages grouped points based on their locations
    averaged_points = []
    for cell_points in grid.values():
        cell_points = np.array(cell_points)
        averaged = cell_points.mean(axis=0)
        averaged_points.append(averaged)

    return np.array(averaged_points).astype(int)


def flag_cosmic_rays(
    images: np.ndarray,
    gauss_stds: np.ndarray = np.array([1000, 1000, 1000]),
    thresh: float = 2e3,
    group_radius: int = 20,
    progress: bool = False,
    debug: bool = False,
) -> tuple[dict, np.ndarray]:
    """
    Masks out low-frequency features using a 3D DFT
    with several masks. This function will then attempt
    to identify the remaining bright pixels and group
    them together if they reside within a user-specified
    distance from each other.

    Parameters:
    -----------
    images :: np.ndarray
        A 3D array of images ordered such that dimensions
        are (time, y, x).
    gauss_stds :: np.ndarray
        A tuple of std values to be used when constructing
        an n-dimensional Gaussian mask. The order of each
        entry should correspond with the respective dimension
        of your image array.
    thresh :: float
        The minimum brightness required to flag a bright
        pixel as a potential cosmic ray. Note that this
        threshold applies to the reconstructed image after
        filtering has been applied.
    group_radius :: int
        The distance (pixels) within which two bright pixels
        will be grouped into a single coordinate at located
        between them.
    progress :: bool
        Toggles progress bar.
    debug :: bool
        Allows for diagnostic plotting.

    Returns:
    --------
    cosmic_ray_locs :: dict
        A dictionary containing the image indexes in which
        cosmic rays were found, along with a list of each
        cosmic ray's location.
    reconstructed_ims :: np.ndarray
        An array of images where low-frequency contributions
        have been damped in the FFT. Has the same shape as
        the input image array.
    """

    # Generates the shifted FFT of an image array
    # NOTE: NaN handling creates sharp, time-persistent features
    dft_shifted = jnp.fft.fftshift(jnp.fft.fftn(jnp.nan_to_num(images, nan=0.0)))

    # De-weights low-frequency contributions in all dimensions
    mask = 1 - _gaussian_mask(
        dft_shifted.shape, tuple(i // 2 for i in images.shape), gauss_stds
    )
    dft_shifted = jnp.multiply(dft_shifted, mask)

    # Inverse DFT back into an n-dimensional image
    reconstructed_ims = jnp.abs(jnp.fft.ifftn(jnp.fft.ifftshift(dft_shifted)))

    # Locates bright pixels and collects them into a dictionary
    cosmic_ray_locs = {}
    if len(images.shape) == 3:
        for idx in tqdm(
            range(len(images)),
            disable=not progress,
        ):
            y_locations, x_locations = np.where(reconstructed_ims[idx] > thresh)
            cosmic_ray_locs[idx] = _cluster_in_grid(
                x_locations, y_locations, group_radius
            )
    else:
        y_locations, x_locations = np.where(reconstructed_ims > thresh)
        cosmic_ray_locs[0] = _cluster_in_grid(x_locations, y_locations, group_radius)

    # Optional plotting
    if debug:

        # Unpacks all (x,y) coordinates for cosmic rays
        x_locs = []
        y_locs = []
        for key in cosmic_ray_locs.keys():
            for x, y in cosmic_ray_locs[key]:
                x_locs.append(x)
                y_locs.append(y)

        # Handles time-series data
        if len(images.shape) == 3:
            xbin_width = len(images[0, 0]) // 25
            ybin_width = len(images[0]) // 25
            xbins = np.arange(0, len(images[0, 0]), xbin_width)
            ybins = np.arange(0, len(images[0]), ybin_width)

        # Handles single-image data
        else:
            xbin_width = len(images[0]) // 25
            ybin_width = len(images) // 25
            xbins = np.arange(0, len(images[0]), xbin_width)
            ybins = np.arange(0, len(images), ybin_width)

        # Initializes figure
        fig, axs = plt.subplot_mosaic(
            [["histx", "."], ["scatter", "histy"]],
            figsize=(8, 8),
            width_ratios=(4, 1),
            height_ratios=(1, 4),
            layout="constrained",
        )

        # Plots scattered data for each cosmic ray location
        axs["scatter"].scatter(x_locs, y_locs, color="k", s=2)
        axs["scatter"].set_xlabel("Dispersion Axis [pixels]")
        axs["scatter"].set_ylabel("Cross-Dispersion Axis [pixels]")
        axs["scatter"].set_xlim(min(xbins), max(xbins))
        axs["scatter"].set_ylim(min(ybins), max(ybins))

        # Plots histogram of horiztonal pixel positions
        axs["histx"].hist(
            x_locs, bins=xbins, density=True, rwidth=0.9, edgecolor="k", color="gray"
        )
        axs["histx"].tick_params(axis="x", labelbottom=False)
        axs["histx"].set_xlim(min(xbins), max(xbins))

        # Plots histogram of vertial pixel positions
        axs["histy"].hist(
            y_locs,
            bins=ybins,
            orientation="horizontal",
            density=True,
            rwidth=0.95,
            edgecolor="k",
            color="gray",
        )
        axs["histy"].tick_params(axis="y", labelleft=False)
        axs["histy"].set_ylim(min(ybins), max(ybins))
        plt.show()

    return cosmic_ray_locs, reconstructed_ims


def verify_cosmic_rays(
    images: np.ndarray,
    locs: dict,
    thresh: float,
    time_kernel_size: int = 10,
    ray_padding: int = 1,
    group_radius: int = 20,
    progress: bool = False,
    debug: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Verifies a handful of user-specified cosmic rays
    by analyzing a small region around the flagged
    pixels(s).

    Parameters:
    -----------
    images :: np.ndarray
        Array of images in which cosmic rays have
        been identified. The ordering of this image
        array should correspond with the entries in
        the 'locs' dictionary.
    locs :: dict
        Has entries of the form:
            {0: [(x1,y1), (x2,y2), ...], 1: [...]}
        The keys should indicate the image index each
        list of points refers to. Each point should
        represent a single cosmic ray where the (x,y)
        coordinate are ints representing its pixel
        position.
    thresh :: float
        The minimum σ-normalized difference to consider
        a confirmed cosmic ray. Pixels matching this
        condition will be represented as 1s in the
        corresponding mask.
    time_kernel_size :: int
        The number of surrounding images to check for
        generating a median exposure. Specifically, the
        median exposure will range from -N//2 to N//2.
    ray_padding :: int
        How many pixels around confirmed cosmic ray pixels
        to also provide a nonzero value in the output mask.
    group_radius :: int
        The distance (pixels) within which two bright pixels
        will be grouped into a single coordinate at located
        between them.
    progress :: bool
        Toggles progress bar.
    debug :: bool
        Allows for optional diagnostic plots.

    Returns:
    --------
    masks :: np.ndarray
        Masks containing 0 for non-contaminated pixels, and
        non-zero entries for potentially-contaminated pixels.
    verified_cosmic_rays :: dict
        A dictionary containing the 'confirmed' cosmic rays
        found in each exposure.
    """

    # Checks the image array dimensions
    single_image = False
    if len(images.shape) == 2:
        images = np.array([images])
        single_image = True

    # Sets up necessary data structures
    verified_cosmic_rays = {}
    masks = []

    # Iterates over each image
    for idx in tqdm(
        range(len(images)),
        disable=not progress,
    ):

        # Initializes data structures for cosmic ray data
        verified_cosmic_rays[idx] = []
        complete_mask = np.zeros(images[0].shape)

        # Iterates over every cosmic ray coordinate
        for point_idx, (x, y) in enumerate(locs[idx]):

            # Defines which images to perform statistics on
            start_idx = max(0, idx - time_kernel_size // 2)
            end_idx = min(len(images), idx + time_kernel_size // 2 + 1)

            # Defines how much space around each cosmic ray should be checked
            xmin, ymin = max(0, x - group_radius), max(0, y - group_radius)
            xmax, ymax = min(len(images[0, 0]), x + group_radius), min(
                len(images[0]), y + group_radius
            )

            # Calculates / pulls the relevant image data for a potential c.r. region
            if single_image:
                sub_im_avg = jnp.ones(
                    images[0, ymin:ymax, xmin:xmax].shape
                ) * jnp.median(images[0, ymin:ymax, xmin:xmax])
                sub_im_std = jnp.ones(
                    images[0, ymin:ymax, xmin:xmax].shape
                ) * jnp.array(mad_std(images[0, ymin:ymax, xmin:xmax]))
                sub_im_raw = jnp.array(images[0, ymin:ymax, xmin:xmax])
            else:
                sub_im_avg = jnp.median(
                    images[start_idx:end_idx, ymin:ymax, xmin:xmax], axis=0
                )
                sub_im_std = jnp.array(
                    mad_std(images[start_idx:end_idx, ymin:ymax, xmin:xmax], axis=0)
                )
                sub_im_raw = jnp.array(images[idx, ymin:ymax, xmin:xmax])

            # Calculates the std-normalized difference
            diff = np.clip((sub_im_raw - sub_im_avg) / sub_im_std, 1e-20, None)

            # Creates a mask for pixels above the user-specified threshold
            mask = jnp.where(diff >= thresh, 1.0, 0.0)

            # Pads the mask according to the user-specified arguments
            mask = _expand_mask(mask, ray_padding=ray_padding)

            # Appends cosmic ray mask data to the relevant arrays
            if jnp.any(mask):
                verified_cosmic_rays[idx].append([(x, y), mask])
            complete_mask[ymin:ymax, xmin:xmax] += mask

            # Optional plotting
            if debug:

                fig, axs = plt.subplot_mosaic("AAAA;BCDE", figsize=(12, 8))

                # Create a rectangle patch
                rect = patches.Rectangle(
                    (x - group_radius, y - group_radius),
                    2 * group_radius,
                    2 * group_radius,
                    edgecolor="red",
                    fill=False,
                )
                axs["A"].add_patch(rect)

                # Plots several of the above images
                axs["A"].imshow(
                    images[idx],
                    cmap="inferno",
                    norm="log",
                    interpolation="none",
                    origin="lower",
                    aspect="auto",
                )
                axs["B"].imshow(
                    sub_im_raw,
                    cmap="inferno",
                    norm="log",
                    interpolation="none",
                    origin="lower",
                )
                axs["C"].imshow(
                    sub_im_avg,
                    cmap="inferno",
                    norm="log",
                    interpolation="none",
                    origin="lower",
                )
                axs["D"].imshow(
                    diff,
                    cmap="inferno",
                    vmin=0,
                    vmax=thresh,
                    interpolation="none",
                    origin="lower",
                )
                axs["E"].imshow(
                    mask,
                    cmap="inferno",
                    vmin=0,
                    vmax=1,
                    interpolation="none",
                    origin="lower",
                )

                # Adds labels to each image
                axs["B"].set_title("Flagged Image")
                axs["C"].set_title("Median Image")
                axs["D"].set_title("σ-Normalized Diff.")
                axs["E"].set_title("Cosmic Ray Mask")

                axs["B"].set_xticks([])
                axs["B"].set_yticks([])
                axs["C"].set_xticks([])
                axs["C"].set_yticks([])
                axs["D"].set_xticks([])
                axs["D"].set_yticks([])
                axs["E"].set_xticks([])
                axs["E"].set_yticks([])

                plt.tight_layout()
                plt.show()

        # Saves composite cosmic ray mask to array
        masks.append(complete_mask)

    return np.array(masks), verified_cosmic_rays


def replace_cosmic_rays(
    images: np.ndarray,
    masks: np.ndarray,
    time_kernel: int = 5,
    progress: bool = False,
) -> np.ndarray:
    """
    Parameters:
    -----------
    images :: np.ndarray
        Raw time-series array of images used
        for averaging out cosmic ray contamination.
    masks :: np.ndarray
        A collection of masks that represet the
        location of cosmic rays along the detector.
        Any non-zero entry in a mask will be converted
        to 1.
    time_kernel :: int
        The number of images to take the median over
        when converting the cosmic ray pixels into
        filtered ones.
    progress :: bool
        Toggles progress bar.

    Returns:
    --------
    filtered_ims :: np.ndarray
        A time-series array where cosmic rays have
        been replaced with the median of surrounding
        exposures at that location.
    """

    # Copies arrays to prevent overwriting data
    filtered_ims = np.array(images).copy()
    masks = np.array(masks).copy()

    # Iterates over every image
    for image_idx in tqdm(
        range(len(images)),
        disable=not progress,
    ):

        # Determiens the closest image indices to use for median calculations
        start_idx = max(0, image_idx - time_kernel // 2)
        end_idx = min(len(filtered_ims), image_idx + time_kernel // 2)

        # Calculates the median image and prepares the cosmic ray mask
        median_image = np.median(filtered_ims[start_idx:end_idx], axis=0)
        mask = masks[image_idx]
        mask[mask > 0] = 1

        # Replaces cosmic ray pixels with the median of surrounding exposures
        filtered_ims[image_idx] = (
            filtered_ims[image_idx] * (1 - mask) + median_image * mask
        )

    return filtered_ims
