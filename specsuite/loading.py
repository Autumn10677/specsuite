import numpy as np
import textwrap
import os
import re

import warnings

from astropy.io import fits
from astropy import coordinates as coord
from astropy.time import Time
import astropy.units as u

# Filled in using registry system
SUPPORTED_INSTRUMENTS = {}


# Simplifies warning to remove visual clutter
def custom_formatwarning(
    message, category, filename=None, lineno=None, line=None, module=None
):
    return f"{category.__name__}: {message}"


warnings.formatwarning = custom_formatwarning


def register_instrument(name):
    def decorator(func):
        SUPPORTED_INSTRUMENTS[name] = func
        return func

    return decorator


def _format_metadata(x: list, test=False) -> np.ndarray | str | bool | float:
    """
    Attempts to convert a list of strings into Numpy array
    with type 'float' or 'bool'. Any lists that cannot be
    converted successfully are kept as a string. Lists with
    one unique value are simply returned as that value.

    Parameters:
    -----------
    x :: list
        A 1D list of strings.

    Returns:
    --------
    x :: np.ndarray | str | bool | float
        A 1D array / single value of type str, bool, or float.
    """

    x = np.array(x)

    # If 'bool' or 'float' conversion fails, default to 'str'
    try:
        if x[0] in ["True", "False"]:
            x = x.astype(bool)
        else:
            x = x.astype(float)
    except Exception:
        x = x.astype(str)

    # Handles repeated / single-entry lists
    if (len(np.unique(x)) == 1) or (len(x) == 1):
        return x[0]

    return x


def _initialize_fits_lists(hdus: list) -> tuple[list, list]:
    """
    Initializes the 'data' and 'metadata' lists used to
    store all FITS file content. Also checks for possible
    incompatibilities bewteen FITS files (i.e., different
    data shapes, missing keywords, etc.) and find a
    resolution to them. If no resolution is possible, returns
    '(None, None)' to tell higher-level functions to stop
    running.

    Parameters:
    -----------
    hdus :: list
        A 1D list of 'astropy.io.fits.hdu.image.PrimaryHDU'
        objects for every exposure.

    Returns:
    --------
    data :: list
        A collection of Numpy arrays with shapes...
            (N_exposure, x_i, y_i)
        ...where 'x_i' and 'y_i' is the shape of the DU content
        for that HDU.
    metadata :: list
        A collection of dictionaries where keys are HU keywords
        and values are 1D lists of metadata. If a 1D value list
        only has one unique value, the list is replaced with the
        single unique value.
    """

    # Helps with readibility below
    N_files = len(hdus)
    N_hdus = len(hdus[0])

    # In-place assignment for efficiency
    data = [None for _ in range(N_hdus)]
    metadata = [None for _ in range(N_hdus)]

    for hdu_idx in range(N_hdus):

        key_sets = [None for _ in hdus]
        data_shapes = [None for _ in hdus]

        for file_idx in range(N_files):

            # Sets are useful for finding whether a list is a subset of another
            key_sets[file_idx] = set(list(hdus[file_idx][hdu_idx].header.keys()))

            # We will use this to look for incompatible shapes
            fits_data = hdus[file_idx][hdu_idx].data
            data_shapes[file_idx] = (-1, -1) if fits_data is None else fits_data.shape

        # Ensures a given DU for all exposures have the same shape
        valid_data_shapes = len(set(data_shapes)) == 1

        # Only 'True' if smaller HUs do not have unique keys
        target_set = max(key_sets, key=len)
        valid_header_shapes = all(s.issubset(target_set) for s in key_sets)

        # Impossible for loading functions to process the data
        if not valid_header_shapes:
            warnings.warn("FITS files have incompatible header shapes...", UserWarning)
            return None, None
        if not valid_data_shapes:
            warnings.warn("FITS files have incompatible data shapes...", UserWarning)
            return None, None

        # Initializes metadata dictionary for later in-place assignment
        metadata[hdu_idx] = {
            str(key): [None for _ in range(N_files)]
            for key in target_set
            if str(key) != ""
        }

    return data, metadata


def _extract_detector_region(string: str) -> tuple[int, int, int, int]:
    """
    A simple helper function that takes strings of
    the form '[#1:#2, #3:#4]' and converts it into
    four integers.

    Parameters:
    -----------
    string :: str
        A string of the form '[#1:#2, #3:#4]'.

    Returns:
    --------
    xstart :: int
        The value of '#1' as an integer.
    xend :: int
        The value of '#2' as an integer.
    ystart :: int
        The value of '#3' as an integer.
    yend :: int
        The value of '#4' as an integer.
    """

    string = re.sub(r"[(){}\[\]]", "", string)
    xstring = string.split(",")[0].split(":")
    ystring = string.split(",")[1].split(":")
    xstart, xend, ystart, yend = (
        int(xstring[0]),
        int(xstring[1]),
        int(ystring[0]),
        int(ystring[1]),
    )

    return xstart, xend, ystart, yend


def _extract_hdu_data(hdus: list) -> tuple[list, list]:
    """
    Extracts all data / metadata into lists of
    arrays / dictionaries. Each list entry corresponds
    to a single FITS HDU unit, and the length of each
    entry corresponds to the number of exposures.

    Parameters:
    -----------
    hdus :: list
        A 1D list of 'astropy.io.fits.hdu.image.PrimaryHDU'
        objects for every exposure.

    Returns:
    --------
    data :: list
        A collection of Numpy arrays with shapes...
            (N_exposure, x_i, y_i)
        ...where 'x_i' and 'y_i' is the shape of the DU content
        for that HDU.
    metadata :: list
        A collection of dictionaries where keys are HU keywords
        and values are 1D lists of metadata. If a 1D value list
        only has one unique value, the list is replaced with the
        single unique value.
    """

    # Helps with readability later on
    N_components = len(hdus[0])

    # Helps us prevent issues when HUs and DUs have mismatches
    data, metadata = _initialize_fits_lists(hdus=hdus)

    # Only triggers if no resolution was possible
    if (data is None) or (metadata is None):
        return None

    for hdu_idx in range(N_components):

        # 'data' cannot be a 'np.ndarray' since entries have non-homogenous shapes
        data[hdu_idx] = np.array([hdu[hdu_idx].data for hdu in hdus])

        # '_initialize_fits_lists()' should have found all possible keys
        valid_keys = list(metadata[hdu_idx].keys())

        # The 'KeyError' should only trigger when a non-vital keyword is missing
        for file_idx, hdu in enumerate(hdus):
            for key in valid_keys:
                try:
                    metadata[hdu_idx][key][file_idx] = hdu[hdu_idx].header[key]
                except KeyError:
                    pass

        # Final pass to fix formatting / type issues in metadata
        metadata[hdu_idx] = {
            k: _format_metadata(metadata[hdu_idx][k]) for k in valid_keys
        }

    # Prevents accidental overwriting
    for hdu in hdus:
        hdu.close()

    return data, metadata


@register_instrument("kosmos")
def _kosmos_formatter(
    data: list,
    metadata: list,
    crop_bds: list,
) -> np.ndarray:
    """
    Data formatter for APO's KOSMOS spectrograph. Information about
    this instrument can be found at...

        https://www.apo.nmsu.edu/mainpage/kosmos/kosmosguide/

    Hard-coded values are utilized when the information is not
    available (or computationally simple to access).

    Parameters:
    -----------
    data :: list
        A list of 3D arrays pulled from the content of FITS files.
        The list should have a length equal to the total number of
        HDUs in each FITS file.
    metadata :: list
        A list of metadata dictionaries pulled from raw FITS files.
        The length of 'metadata' should be the same as 'data'.
    crop_bds :: list


    Returns:
    --------
    formatted_data :: np.ndarray
        A 3D Numpy array of the shape...
            (N_exposures, x_wavelength, y_spatial)
    """

    # Defined here since it is not defined in the FITS metadata
    GAIN = 0.6

    # Unpacked here since files only have one HDU
    data = data[0] * GAIN
    metadata = metadata[0]

    # Aligns images so x = dispersion and y = cross-dispersion
    data = np.rot90(data, k=3, axes=(1, 2))

    # Just clipping (no subtraction) due to poor charge transfer efficiency
    overscan_start, _, _, _ = _extract_detector_region(metadata["BSEC11"])
    formatted_data = data[:, :overscan_start]

    formatted_data = formatted_data[:, crop_bds[0] : crop_bds[1]]

    metadata["RN"] = 6

    return formatted_data, metadata


@register_instrument("gmos-hamamatsu")
def _gmos_hamamatsu_formatter(
    data: list,
    metadata: list,
    crop_bds: list,
) -> np.ndarray:
    """
    Data formatter for GMOS' e2v DD (pre-2017) detector layout.
    As of now, this function assumes data was collected in
    '6-amp' mode described at...

        https://www.gemini.edu/instrumentation/gmos/data-reduction

    Hard-coded values are utilized when the information is not
    available (or computationally simple to access). The chip gap
    is in units of 'pixels' and is adjusted for various binnings.
    Regardless of binning, overscan regions should always be 32
    pixels wide.

    Parameters:
    -----------
    data :: list
        A list of 3D arrays pulled from the content of FITS files.
        The list should have a length equal to the total number of
        HDUs in each FITS file.
    metadata :: list
        A list of metadata dictionaries pulled from raw FITS files.
        The length of 'metadata' should be the same as 'data'.
    crop_bds :: list
        The region along the cross-dispersion (spatial) axis
        to keep (all other rows will be dropped).

    Returns:
    --------
    combined_data :: np.ndarray
        A 3D Numpy array of the shape...
            (N_exposures, x_wavelength, y_spatial)
    """

    # Assuming 12-amp mode
    CHIP_ORDER = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    # Tables pulled from ee2v DD documentation page
    GAINS = {
        "Normal Science": [
            1.66,
            1.63,
            1.62,
            1.57,
            1.68,
            1.65,
            1.64,
            1.68,
            1.61,
            1.63,
            1.58,
            1.65,
        ],
        "Acquisition": [
            2.01,
            1.95,
            1.97,
            1.91,
            2.03,
            1.97,
            1.96,
            2.01,
            1.95,
            1.95,
            1.89,
            1.95,
        ],
        "Bright Target": [
            5.21,
            5.18,
            5.05,
            5.04,
            5.29,
            5.13,
            5.14,
            5.29,
            5.03,
            5.10,
            4.86,
            5.09,
        ],
    }
    READ_NOISES = {
        "Normal Science": [
            4.06,
            4.12,
            4.12,
            3.99,
            4.20,
            3.98,
            3.88,
            4.20,
            4.04,
            4.35,
            4.02,
            4.55,
        ],
        "Acquisition": [
            8.02,
            6.88,
            6.01,
            5.88,
            6.71,
            6.34,
            5.39,
            5.86,
            6.10,
            6.13,
            5.98,
            6.79,
        ],
        "Bright Target": [
            9.90,
            8.94,
            8.75,
            8.17,
            9.42,
            8.91,
            8.19,
            8.84,
            8.13,
            8.53,
            8.03,
            8.80,
        ],
    }

    # Necessary for figuring out which instrument mode was used
    avg_gain = 0.0
    avg_rn = 0.0
    for entry in metadata[1:]:
        avg_gain += entry["GAIN"] / (len(metadata) - 1)
        avg_rn += entry["RDNOISE"] / (len(metadata) - 1)

    # Values seem arbitrary, but are based on web-based documentation
    if (avg_gain < 1.8) and (avg_rn < 5):
        MODE = "Normal Science"
    elif (avg_gain < 4) and (avg_rn < 7):
        MODE = "Acquisition"
    else:
        MODE = "Bright Target"

    # Defined here to improve readibility later on
    N_COLS = 12
    N_ROWS = int((len(data) - 1) / N_COLS)
    N_EXPOSURES = len(data[0])

    # By default, this is a string (i.e., '2 2')
    BINNING = (
        int(metadata[1]["CCDSUM"][0]),
        int(metadata[1]["CCDSUM"][-1]),
    )

    # Adjusts for bin size's impact on effect gap width
    CHIP_GAP = 61
    CHIP_GAP = CHIP_GAP // BINNING[0]

    # Overscan length should be unaffected by binning
    CONTAM_PIX_LENGTH = 5

    # Their 'X' and 'Y' conventions are flipped from ours
    XSHAPE = metadata[0]["DETRO1XS"] + 2 * CHIP_GAP
    YSHAPE = 0
    for i in range(N_ROWS):
        YSHAPE += metadata[0][f"DETRO{i + 1}YS"]

    # Using Numpy array stacking can eat huge chunks of memory
    combined_data = np.full(
        (
            N_EXPOSURES,
            int(YSHAPE),
            int(XSHAPE),
        ),
        np.nan,
    )
    # This will be stored in the metadata dictionary
    combined_RN = np.full(
        (
            int(YSHAPE),
            int(XSHAPE),
        ),
        np.nan,
    )

    y_offset = 0

    for i in range(N_ROWS):
        for j in range(N_COLS):

            # Pulls from the gain / RN tables for that chip / mode
            GAIN = GAINS[MODE][j]
            RN = READ_NOISES[MODE][j]

            # Convenient for readibility later on
            REAL_IDX = CHIP_ORDER[j] + N_COLS * i

            # Extracts overscan / data regions from metadata strings
            over_start, over_end, _, _ = _extract_detector_region(
                metadata[REAL_IDX]["BIASSEC"]
            )
            data_start, data_end, ystart, yend = _extract_detector_region(
                metadata[REAL_IDX]["DATASEC"]
            )

            # Used for calculating where data from each chip slots in
            DATALEN = int(data_end - data_start + 1)
            DATA_YLEN = int(yend - ystart + 1)

            # Two cases exist since overscan is sometimes on left/right of data
            if j % 2 == 1:
                over_end -= CONTAM_PIX_LENGTH
            else:
                over_start += CONTAM_PIX_LENGTH

            # Really nasty, but this is quick and helps with readibility later!
            gap_offset = CHIP_GAP * (((i * N_COLS + j) // 4) % 3)
            total_offset = j * DATALEN + gap_offset

            # Creates a 3D array from a 2D median overscan array
            bias = np.repeat(
                np.median(
                    data[REAL_IDX][:, :, over_start - 1 : over_end],
                    axis=2,
                )[:, :, np.newaxis],
                DATALEN,
                axis=2,
            )

            # Clips out the overscan region for that chip
            chip_data = data[REAL_IDX][:, :, data_start - 1 : data_end]

            # Inserts all data for that chip into the 3D / 2D arrays
            combined_data[
                :,
                y_offset : y_offset + DATA_YLEN,
                total_offset : total_offset + DATALEN,
            ] = (chip_data - bias) * GAIN
            combined_RN[
                y_offset : y_offset + DATA_YLEN,
                total_offset : total_offset + DATALEN,
            ] = RN

        y_offset += DATA_YLEN

    # Easiest to rotate the image here
    combined_data = np.rot90(combined_data, k=2, axes=(1, 2))
    combined_data = combined_data[:, crop_bds[0] : crop_bds[1]]
    combined_RN = np.rot90(combined_RN, k=2)
    combined_RN = combined_RN[crop_bds[0] : crop_bds[1]]

    # Doing this to avoid having optional third return
    metadata[0]["RN"] = combined_RN

    return combined_data, metadata


@register_instrument("gmos-e2vDD")
def _gmos_e2vDD_formatter(
    data: list,
    metadata: list,
    crop_bds: list,
) -> np.ndarray:
    """
    Data formatter for GMOS' e2v DD (pre-2017) detector layout.
    As of now, this function assumes data was collected in
    '6-amp' mode described at...

        https://www.gemini.edu/instrumentation/gmos/data-reduction

    Hard-coded values are utilized when the information is not
    available (or computationally simple to access). The chip gap
    is in units of 'pixels' and is adjusted for various binnings.
    Regardless of binning, overscan regions should always be 32
    pixels wide.

    Parameters:
    -----------
    data :: list
        A list of 3D arrays pulled from the content of FITS files.
        The list should have a length equal to the total number of
        HDUs in each FITS file.
    metadata :: list
        A list of metadata dictionaries pulled from raw FITS files.
        The length of 'metadata' should be the same as 'data'.
    crop_bds :: list
        The region along the cross-dispersion (spatial) axis
        to keep (all other rows will be dropped).

    Returns:
    --------
    combined_data :: np.ndarray
        A 3D Numpy array of the shape...
            (N_exposures, x_wavelength, y_spatial)
    """

    # Assuming 6-amp mode
    CHIP_ORDER = np.array([2, 1, 4, 3, 5, 6])

    # Tables pulled from ee2v DD documentation page
    GAINS = {
        "low_slow": [2.31, 2.31, 2.21, 2.27, 2.17, 2.33],
        "low_fast": [2.53, 2.53, 2.41, 2.50, 2.43, 2.56],
        "high_fast": [5.36, 5.34, 5.10, 5.27, 5.16, 5.41],
    }
    READ_NOISES = {
        "low_slow": [3.41, 3.17, 3.20, 3.22, 3.46, 3.44],
        "low_fast": [4.3, 4.2, 4.1, 4.1, 4.3, 4.4],
        "high_fast": [7.0, 7.8, 5.7, 5.8, 6.4, 6.3],
    }

    # Value of 5000 represents 1000
    AMP_MODE = metadata[0]["AMPINTEG"]
    AMP_MODE = "slow" if AMP_MODE == 1000 else "fast"

    # 3.0 threshold pulled from average RN for 'slow' mode
    GAIN_MODE = 0
    for idx in range(1, len(metadata)):
        GAIN_MODE += metadata[idx]["GAIN"] / (len(metadata) - 1)
    GAIN_MODE = "low" if GAIN_MODE < 3.0 else "high"

    # Defined here to improve readibility later on
    N_COLS = len(CHIP_ORDER)
    N_ROWS = int((len(data) - 1) / 6)
    N_EXPOSURES = len(data[0])

    # By default, this is a string (i.e., '2 2')
    BINNING = (
        int(metadata[1]["CCDSUM"][0]),
        int(metadata[1]["CCDSUM"][-1]),
    )

    # Adjusts for bin size's impact on effect gap width
    CHIP_GAP = 39
    CHIP_GAP = CHIP_GAP // BINNING[0]

    # Overscan length should be unaffected by binning
    CONTAM_PIX_LENGTH = 5

    # Their 'X' and 'Y' conventions are flipped from ours
    XSHAPE = metadata[0]["DETRO1XS"] + 2 * CHIP_GAP
    YSHAPE = 0
    for i in range(N_ROWS):
        YSHAPE += metadata[0][f"DETRO{i + 1}YS"]

    # Using Numpy array stacking can eat huge chunks of memory
    combined_data = np.full(
        (
            N_EXPOSURES,
            int(YSHAPE),
            int(XSHAPE),
        ),
        np.nan,
    )
    # This will be stored in the metadata dictionary
    combined_RN = np.full(
        (
            int(YSHAPE),
            int(XSHAPE),
        ),
        np.nan,
    )

    y_offset = 0

    for i in range(N_ROWS):
        for j in range(N_COLS):

            # Pulls from the gain / RN tables for that chip / mode
            GAIN = GAINS[f"{GAIN_MODE}_{AMP_MODE}"][j]
            RN = READ_NOISES[f"{GAIN_MODE}_{AMP_MODE}"][j]

            # Convenient for readibility later on
            REAL_IDX = CHIP_ORDER[j] + N_COLS * i

            # Extracts overscan / data regions from metadata strings
            over_start, over_end, _, _ = _extract_detector_region(
                metadata[REAL_IDX]["BIASSEC"]
            )
            data_start, data_end, ystart, yend = _extract_detector_region(
                metadata[REAL_IDX]["DATASEC"]
            )

            # Used for calculating where data from each chip slots in
            DATALEN = int(data_end - data_start + 1)
            DATA_YLEN = int(yend - ystart + 1)

            # Chip order is weird, so using metadata to locate overscan
            AMPNAME = metadata[REAL_IDX]["AMPNAME"]
            if "right" in AMPNAME:
                over_end -= CONTAM_PIX_LENGTH
            else:
                over_start += CONTAM_PIX_LENGTH

            # Really nasty, but this is quick and helps with readibility later!
            gap_offset = CHIP_GAP * (((i * N_COLS + j) // 2) % 3)
            total_offset = j * DATALEN + gap_offset

            # Creates a 3D array from a 2D median overscan array
            bias = np.repeat(
                np.median(
                    data[REAL_IDX][:, :, over_start - 1 : over_end],
                    axis=2,
                )[:, :, np.newaxis],
                DATALEN,
                axis=2,
            )

            # Clips out the overscan region for that chip
            chip_data = data[REAL_IDX][:, :, data_start - 1 : data_end]

            # Inserts all data for that chip into the 3D / 2D arrays
            combined_data[
                :,
                y_offset : y_offset + DATA_YLEN,
                total_offset : total_offset + DATALEN,
            ] = (chip_data - bias) * GAIN
            combined_RN[
                y_offset : y_offset + DATA_YLEN,
                total_offset : total_offset + DATALEN,
            ] = RN

        y_offset += DATA_YLEN

    # Easiest to rotate the image here
    combined_data = np.rot90(combined_data, k=2, axes=(1, 2))
    combined_data = combined_data[:, crop_bds[0] : crop_bds[1]]
    combined_RN = np.rot90(combined_RN, k=2)
    combined_RN = combined_RN[crop_bds[0] : crop_bds[1]]

    # Doing this to avoid having optional third return
    metadata[0]["RN"] = combined_RN

    return combined_data, metadata


def _format_data(
    data: list,
    metadata: list,
    instrument: str,
    crop_bds: list,
) -> tuple[list, list] | tuple[np.ndarray, dict]:
    """
    Handles formatting 'DU' and 'HU' lists into more user-friendly
    formats. If the provided 'instrument' is not supported,
    both 'data' and 'metadata' are returned unaltered.

    Parameters:
    -----------
    data :: list
        A list of 3D arrays pulled from the content of FITS files.
        The list should have a length equal to the total number of
        HDUs in each FITS file.
    metadata :: list
        A list of metadata dictionaries pulled from raw FITS files.
        The length of 'metadata' should be the same as 'data'.
    instrument :: str
        The name of the instrument the FITS data was
        taken from. This is used to determine which formatting
        function should be used.
    crop_bds :: list
        The region along the cross-dispersion (spatial) axis
        to keep (all other rows will be dropped).

    Returns:
    --------
    data :: list | np.ndarray
        The formatted data arrays.
    metadata :: list | dict
        The formatted metadata dictionaries.
    """

    # 'SUPPORTED_INSTRUMENTS' has 'values' pointing to functions
    try:
        return SUPPORTED_INSTRUMENTS[instrument](
            data=data,
            metadata=metadata,
            crop_bds=crop_bds,
        )

    # Hopefully only trigger for 'default' instrument
    except KeyError:
        return data, metadata


def filter_files(files: list, tag: str, ignore: list):
    """
    Filters down a list of filenames if they to
    not satisfy the following requirements...

        1) The file ends with '.fits' extension
        2) The provided 'tag' is not in the filename
        3) The filename is not given in 'ignore' list

    Parameters:
    -----------
    files :: list
        Several filenames to filter based on the above
        criteria.
    tag :: str
        A sub-string that can help differentiate between
        desired and undesired files in a directory. If
        an empty string is provided, no files are filtered
        out (based on the 'tag' criteria).
    ignore :: list
        Filenames to ignore when loading in data. The 'ignore'
        filenames must exactly match how they appear in the
        file navigator (including .fits extension).

    Returns:
    --------
    files :: list
        All remaining files once filtering has been performed.
    """

    files = sorted([f for f in files if f.endswith(".fits")])
    files = [f for f in files if ((tag in f) and (f not in ignore))]

    return files


def collect_images_array(
    path: str,
    tag: str = "",
    ignore: list = [],
    crop_bds: list = [0, None],
    instrument: str = "kosmos",
    return_metadata: bool = False,
    debug: bool = False,
) -> tuple[list, list] | tuple[np.ndarray, dict]:
    """
    Collect a list of images from a user-given path
    corresponding to a specified tag. Images can
    be ignore by passing their indexes as an additional
    argument.

    Parameters:
    -----------
    path :: str
        Path to data directory containing image
        data.
    tag :: str
        Tag to search for in filenames.
    ignore :: list
        List of filenames to ignore.
    crop_bds :: list
        The region along the cross-dispersion (spatial) axis
        to keep (all other rows will be dropped).
    instrument :: str
        The name of the instrument the FITS data was
        taken from. This is used to determine which formatting
        function should be used.
    return_metadata :: bool
        Toggles whether metadata is return alongside file data.
    debug :: bool
        Allows for diagnostic information to be printed.
        This includes the names of all files found with
        the given 'tag' and whether any of them failed
        to load.

    Returns:
    --------
    data :: list | np.ndarray
        The formatted data arrays.
    metadata :: list | dict
        The formatted metadata dictionaries.
    """

    # Informs user that their instrument name was not recognized
    if instrument not in SUPPORTED_INSTRUMENTS:
        warnings.warn(
            f"'{instrument}' is not a supported instrument...\n - "
            + "\n - ".join(SUPPORTED_INSTRUMENTS.keys())
            + "\nUsing the 'default' loading procedure!"
        )
        instrument = "default"

    # Attempts to load and filter filenames
    try:
        files = filter_files(os.listdir(path), tag, ignore)
    except NotADirectoryError:
        warnings.warn(
            f"The provided directory '{path}' is not a valid path", UserWarning
        )
        return None

    # Tells the user if no files were found to prevent confusion over 'None'
    if len(files) == 0:
        warnings.warn(
            f"No files in '{path}' with tag '{tag}' were found...", UserWarning
        )
        return None

    hdus = []

    # Printed here to allow the following loop
    if debug:
        print(f"\nSearching for files with '{tag}' tag...")
        print("------------------------------------------")

    # Iterating in this way to allow errors to show in 'debug' mode
    for f in files:

        # Hopefully only fails when file is not a valid '.fits' file
        try:
            hdus.append(fits.open(os.path.join(path, f)))
            hdus[-1].verify("silentfix")
            if debug:
                print(f"  ✓ {f}")

        # 'textwrap' ensures the error message isn't too long
        except Exception as e:
            if debug:
                print(
                    f"  X {f}\n      --> "
                    + textwrap.fill(str(e), width=60, subsequent_indent="          ")
                    + "\n"
                )

    # Should generalize to any instrument
    data, metadata = _extract_hdu_data(
        hdus=hdus,
    )
    if (data is None) or (metadata is None):
        return None

    # If 'default', 'data' and 'metadata' are unchanged
    data, metadata = _format_data(
        data=data,
        metadata=metadata,
        instrument=instrument,
        crop_bds=crop_bds,
    )

    # Only returns metadata if strictly requested
    if return_metadata:
        return data, metadata
    return data


def average_matching_files(
    path: str,
    tag: str = "",
    instrument: str = "kosmos",
    ignore: list = [],
    crop_bds: list = [0, None],
    mode: str = "median",
    return_metadata: bool = False,
    debug: bool = False,
) -> np.ndarray:
    """
    Extracts images from a user-given path, and finds
    the average pixel value for every pixel across all
    images. This defaults to the 'median' average, but
    can be changed to take the 'mean' average as well.

    Parameters:
    -----------
    path :: str
        Path to data directory.
    tag :: str
        Tag to search for in filenames.
    instrument :: str
        The name of the instrument your FITS data was taken from. This
        is only used to determine which loading function to use.
    ignore :: list
        List of data indexes to ignore in averaging.
    crop_bds :: list
        The region along the cross-dispersion (spatial) axis
        to keep (all other rows will be dropped).
    mode :: str
        Type of average to take of images. Valid inputs
        include 'median' and 'mean'.
    return_metadata :: bool
        Toggles whether metadata is return alongside file data.
    debug :: bool
        Toggles the display of image stats.
    """

    # Retrieves all data filenames and prepares image list
    content = collect_images_array(
        path,
        tag,
        instrument=instrument,
        ignore=ignore,
        crop_bds=crop_bds,
        return_metadata=True,
        debug=debug,
    )

    # Since some errors return 'None', we check for that here
    if content is None:
        return content
    images, metadata = content

    # Handles 'None' return from 'collect_images_array()'
    try:
        if mode.lower() == "mean":
            avg_image = np.mean(images, axis=0)
        else:
            avg_image = np.median(images, axis=0)
    except np.exceptions.AxisError:
        return None

    # Prints image statistics
    if debug:
        print(f"\nImage statistics for average '{tag}' image...")
        print(rf"      Min: {np.nanmin(avg_image.flatten())}")
        print(rf"      Max: {np.nanmax(avg_image.flatten())}")
        print(rf"     Mean: {round(np.nanmean(avg_image.flatten()), 3)}")
        print(rf"      STD: {round(np.nanstd(avg_image.flatten()), 3)}")

    if return_metadata:
        return avg_image, metadata
    return avg_image


def load_metadata(
    path: str,
    tag: str,
    ignore: list = [],
) -> dict:
    """
    Loads an dictionary of all data for
    a collection of FITS files. This
    metadata comes from the header of the
    first FITS card.

    Parameters:
    -----------
    path :: str
        Path to data directory.
    tag :: str
        Tag to search for in filenames.
    ignore :: list
        List of data indexes to ignore.

    Returns:
    --------
    metadata :: dict
        Dictionary containing the metadata
        found for each key in the FITS headers.
        Keys-value pairs that are identical
        across all exposures are combined into
        a single value.
    """

    # Loads all FITS headers
    files = filter_files(os.listdir(path), tag, ignore)
    adds = [os.path.join(path, file) for file in files]
    hdrs = [fits.open(add)[0].header for add in adds]

    # Extracts all metadata into a dictionary
    metadata = {key: [] for key in hdrs[0].keys()}
    for key in metadata.keys():
        for hdr in hdrs:
            metadata[key].append(hdr[key])

    # Reduces non-unique lists to a single value
    metadata = {
        key: (
            np.array(metadata[key])
            if len(np.unique(metadata[key])) != 1
            else metadata[key][0]
        )
        for key in metadata.keys()
    }

    # Places single-value entries at the front of the dictionary
    list_keys = [
        key for key, value in metadata.items() if isinstance(value, np.ndarray)
    ]
    non_list_keys = [
        key for key, value in metadata.items() if not isinstance(value, np.ndarray)
    ]
    ordered_keys = non_list_keys + list_keys
    metadata = {key: metadata[key] for key in ordered_keys}

    return metadata


def extract_times(
    path: str,
    tag: str,
    ignore: list = [],
    time_lbl: str = "DATE-OBS",
    ra_lbl: str = "RA",
    dec_lbl: str = "DEC",
    lat_lbl: str = "LATITUDE",
    long_lbl: str = "LONGITUD",
    time_format: str = "isot",
    time_scale: str = "tai",
    loc_units: tuple = (u.hourangle, u.deg),
    loc_frame: str = "icrs",
):
    """
    Extracts time data from the headers of a set of
    observations. Assumes that the header has information
    about the observation time.

    Parameters:
    -----------
    path :: str
        Directory pointing toward the FITS file you wish to
        load. This should not include the name of the file
        itself.
    tag :: str
        A sub-string that can help differentiate between
        desired and undesired files in a directory. If
        an empty string is provided, no files are filtered
        out (based on the 'tag' criteria).
    ignore :: list
        Filenames to ignore when loading in data. The 'ignore'
        filenames must exactly match how they appear in the
        file navigator (including .fits extension).
    time_lbl :: str
        Header label for observation time.
    ra_lbl :: str
        Header label for RA of target.
    dec_lbl :: str
        Header label for DEC of target.
    lat_lbl :: str
        Header label for latitude of target.
    long_lbl :: str
        Header label for longitude of target.
    time_format :: str
        Astropy Time() format that represents the
        time data in the header.
    time_scale :: str
        Astropy Time() scale that represents the
        time data in the header.
    loc_units :: tuple
        Astropy SkyCoord() units that represents
        the (RA, DEC) data in the header.
    loc_frame :: str
        Astropy SkyCoord() frame that represents
        the (RA, DEC) data in the header.

    Returns:
    --------
    times_bc :: np.ndarray
        Array of JD barycentric times that have
        been corrected for variations in light
        travel time. Has attached astropy units
        of days.
    """

    # Gets a list of file addresses for our data
    files = sorted(
        [file for file in os.listdir(path) if tag in file and file[-9:-5] not in ignore]
    )
    adds = [os.path.join(path, file) for file in files]

    # Extracts all headers
    hdrs = [fits.open(add)[0].header for add in adds]

    # Extracts relevant data from headers
    times = [hdr[time_lbl] for hdr in hdrs]
    ra = hdrs[0][ra_lbl]
    dec = hdrs[0][dec_lbl]
    latitude = hdrs[0][lat_lbl]
    longitude = hdrs[0][long_lbl]

    # Pulls locations data
    ip_peg = coord.SkyCoord(ra, dec, unit=loc_units, frame=loc_frame)
    location = coord.EarthLocation(lat=latitude, lon=longitude)

    # Calculates two relevant types of time
    times = Time(times, format=time_format, scale=time_scale, location=location)
    ltt_bary = [t.light_travel_time(ip_peg) for t in times]

    # Calculates corrected barycentric times
    times_bc = np.array([(t.tdb + tb).jd for t, tb in zip(times, ltt_bary)]) * u.day

    return times_bc


def split_chips(images: np.ndarray) -> np.ndarray:
    """
    Attempts to split up a series of 2D images into separate
    arrays for each "chip" that has been combined. This
    function assumes that "chip gaps" are indicated by a column
    that is entirely comprised of NaN values.

    Parameters:
    -----------
    images :: np.ndarray
        A series of images that are comprised of multiple chips
        joined by a "chip gap" comprised of NaN values.

    Returns:
    --------
    sub_images :: np.ndarray
        A list of images where each entry has N sub-images that
        make up each chip that was detected.
    """

    # Ensures that code runs for a single image
    if len(images.shape) == 2:
        images = np.array([images])

    sub_images = []

    for image in images:
        nan_cols = np.all(np.isnan(image), axis=0)
        split_idx = np.where(nan_cols[:-1] != nan_cols[1:])[0] + 1
        chips = [
            block
            for block in np.split(image, split_idx, axis=1)
            if not np.all(np.isnan(block))
        ]
        sub_images.append(chips)

    return np.array(sub_images)
