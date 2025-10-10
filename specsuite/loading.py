import numpy as np
import os
import re

import warnings

from tqdm import tqdm
from astropy.io import fits
from astropy import coordinates as coord
from astropy.time import Time
import astropy.units as u

SUPPORTED_INSTRUMENTS = ["kosmos"]


def custom_formatwarning(
    message, category, filename=None, lineno=None, line=None, module=None
):
    return f"{category.__name__}: {message}"


warnings.formatwarning = custom_formatwarning


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


def extract_image(
    path: str,
    file: str,
    instrument: str,
):
    """
    Attempts to extract image data from a given FITS file
    using a method specific to the user-specified instrument.

    Parameters:
    -----------
    path :: str
        Directory pointing toward the FITS file you wish to
        load. This should not include the name of the file
        itself.
    file :: str
        Name of the FITS file in the specified directory to
        load.
    instrument :: str
        Specifies which loading function should be used for
        the FITS file. Currently, the only supported instruments
        are...
            - KOSMOS

    Returns:
        ???
    """

    if instrument == "kosmos":
        return _kosmos_loader(path, file)


def _kosmos_loader(
    path: str,
    file: str,
    clip_overscan: bool = True,
):
    """
    Controls how to load data from Apache Point Observatory's
    KOSMOS instrument. The resulting output will be oriented
    such that the x-axis is the dispersion axis (left is blue /
    right is red) and the y-axis is the cross-dispersion axis.

    Parameters:
    -----------
    path :: str
        Directory pointing toward the FITS file you wish to
        load. This should not include the name of the file
        itself.
    file :: str
        Name of the FITS file in the specified directory to
        load.
    """

    # Extracts header from fits file
    hdu = fits.open(path + f"/{file}")
    image_data = hdu[0].data

    if clip_overscan:

        # Loads metadata for the size of half the overscan region
        bias_section = (hdu[0].header)["BSEC11"]
        bias_section = re.sub(r"[(){}\[\]]", "", bias_section)

        # Calculates the total overscan region length (accounts for indexing)
        overscan_region = bias_section.split(",")[0].split(":")
        overscan_length = int(overscan_region[1]) - int(overscan_region[0])
        overscan_length = 2 * (overscan_length + 1)

        # Rotates the image to make the x-axis our dispersion axis
        image_data = np.rot90(image_data, k=3)
        image_data = image_data[: len(image_data[0]) - overscan_length, :]

    else:
        image_data = np.rot90(image_data, k=3)

    return image_data


def collect_images_array(
    path: str,
    tag: str,
    ignore: list = None,
    crop_bds: list = [0, None],
    instrument: str = "kosmos",
    clip_overscan: bool = True,
    debug: bool = False,
    progress: bool = False,
):
    """
    Collect a list of images from a user-given path
    corresponding to a specified tag. Images can
    be ignore by passing their indexes as an additional
    argument.

    Parameters:
    -----------
    path :: string
        Path to data directory containing image
        data.
    tag :: string
        Tag to search for in filenames.
    ignore :: list
        List of file indexes to ignore.

    Returns:
    --------
    image_collection :: list
        List of np.ndarray objects for each image
        in the user-given file path.
    """

    instrument = instrument.lower()

    if ignore is None:
        ignore = []

    # Prevents users from loading data if the relevant function does not exist
    if instrument not in SUPPORTED_INSTRUMENTS:
        warnings.warn(f"'{instrument}' is not a supported instrument", UserWarning)
        return None

    # Attempts to load and filter filenames
    try:
        files = filter_files(os.listdir(path), tag, ignore)
    except NotADirectoryError:
        warnings.warn(
            f"The provided directory '{path}' is not a valid path", UserWarning
        )
        return None

    if debug:
        print(f"\nSearching for files with '{tag}' tag...")
        print("------------------------------------------")

    # Adds each valid file to list of data
    image_collection = []
    for file in tqdm(files, desc="collecting image array", disable=not progress):

        # Extracts and appends image data
        try:
            image_collection.append(extract_image(path, file, instrument))
            if debug:
                print(f"{file}")
        except Exception as e:
            print(e)
            if debug:
                print(f"{file} --> (FAILED, {e})")

    return np.array(image_collection)[:, crop_bds[0] : crop_bds[1]]


def average_matching_files(
    path: str,
    tag: str,
    ignore: list = [],
    crop_bds: list = [0, None],
    mode: str = "median",
    debug: bool = False,
    progress: bool = False,
):
    """
    Extracts images from a user-given path, and finds
    the average images calculated on a pixel-by-pixel
    basis.

    Parameters:
    -----------
    path :: string
        Path to data directory.
    tag :: string
        Tag to search for in filenames.
    ignore :: list
        List of data indexes to ignore in averaging.
    mode :: str
        Type of average to take of images. Valid inputs
        include 'median' and 'mean'.
    debug :: bool
        Toggles the display of image stats.
    progress :: bool
        Toggles the progress bar.
    """

    # Retrieves all data filenames and prepares image list
    images = []
    images = collect_images_array(path, tag, ignore=ignore, debug=debug, progress=progress)

    mode = mode.lower()
    if mode == "mean":
        avg_image = np.mean(images, axis=0)
    else:
        avg_image = np.median(images, axis=0)

    # Prints image statistics
    if debug:
        print(f"\nImage statistics for average '{tag}' image...")
        print(rf"      Min: {np.min(avg_image.flatten())}")
        print(rf"      Max: {np.max(avg_image.flatten())}")
        print(rf"     Mean: {round(np.mean(avg_image.flatten()), 3)}")
        print(rf"      STD: {round(np.std(avg_image.flatten()), 3)}")

    return avg_image[crop_bds[0] : crop_bds[1]]


def load_metadata(
    path: str,
    tag: str,
    ignore: list = [],
):
    """
    Loads an dictionary of all data for
    a collection of FITS files. This
    metadata comes from the header of the
    first FITS card.

    Parameters:
    -----------
    path :: string
        Path to data directory.
    tag :: string
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
    path,
    target,
    ignore=[],
    time_lbl="DATE-OBS",
    ra_lbl="RA",
    dec_lbl="DEC",
    lat_lbl="LATITUDE",
    long_lbl="LONGITUD",
    time_format="isot",
    time_scale="tai",
    loc_units=(u.hourangle, u.deg),
    loc_frame="icrs",
):
    """
    Extracts time data from the headers of a set of
    observations. Assumes that the header has information
    about the observation time.

    Parameters:
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
    times_bc :: np.ndarray
        Array of JD barycentric times that have
        been corrected for variations in light
        travel time. Has attached astropy units
        of days.
    """

    # Gets a list of file addresses for our data
    files = sorted(
        [
            file
            for file in os.listdir(path)
            if target in file and file[-9:-5] not in ignore
        ]
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
