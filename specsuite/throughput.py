import pandas as pd
import numpy as np
import requests
from io import StringIO
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.units import Quantity
import warnings

import sys

sys.tracebacklimit = 0


# Simplifies warning to remove visual clutter
def custom_formatwarning(
    message, category, filename=None, lineno=None, line=None, module=None
):
    return f"{category.__name__}: {message}"


warnings.formatwarning = custom_formatwarning


def load_STIS_spectra(
    name: str = None,
    filetype: str = "model",
    wavelength_bounds: tuple = None,
    debug: bool = False,
):
    """
    Attempts to download spectra data from the STIS website (see url
    below). It only looks for data contained in the first data table.

    Parameters:
    -----------
    name :: str
        Name of the star to load data for. This should match an entry
        in the "Star name" column of Table 1.
    filetype :: str
        Determines which type of model to load from the STIS database.
        The only valid options are "model" or "stis".
    wavelength_bounds :: tuple
        The (wmin, wmax) region of the STIS spectra to keep. Both
        values must have astropy units compatible with wavelength.
    debug :: bool
        Allows diagnostic information to be output.

    Returns:
    --------
    wavs :: np.ndarray
        Retrieved model wavelengths (Angstroms)
    flux :: np.ndarray
        Retrieved model flux (flam)
    cont :: np.ndarray
        Retrieved model flux (flam)
    """

    # Loads the HTML data from the STIS website
    url = "https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec"  # noqa: E501

    response = requests.get(url)
    df = pd.read_html(StringIO(response.text))[0]
    df = df[df["Star name"] == name].reset_index(drop=True)
    df = df.rename(columns={"STIS**": "stis", "Model": "model"})

    # Prevents an attempted download for a file that does not exist
    assert len(df) > 0, f"Name '{name}' not found in table..."
    assert filetype in [
        "model",
        "stis",
    ], f"filetype must be 'model' or 'stis,' not '{filetype}'"

    if filetype == "model":
        assert (
            type(df["model"][0]) is str
        ), f"'{name}' exists, but no '_model' file exists..."
    else:
        assert (
            type(df["stis"][0]) is str
        ), f"'{name}' exists, but no '_stis' file exists..."

    # Loads FITS data for the specified star
    filename = f"{df["Name"][0]}{df[filetype][0]}.fits"
    file_url = f"https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/{filename}"  # noqa: E501
    hdul = fits.open(file_url)

    # Unpacks spectral data
    data = hdul[1].data
    hdul.close()

    wavs = data["WAVELENGTH"] * u.AA

    # Generates mask for undesired wavelengths
    try:
        if wavelength_bounds is None:
            wavelength_bounds = [np.min(wavs), np.max(wavs)]
        mask = (wavelength_bounds[0] < wavs) & (wavs < wavelength_bounds[1])
        wavs = wavs[mask]
    except (TypeError, u.UnitConversionError):
        print(
            f"Wavelength bounds must be astropy.Quantities, not '{type(wavelength_bounds)}'"  # noqa: E501
        )
        return None

    if filetype == "model":
        cont = data["CONTINUUM"][mask] * u.erg / u.s / u.cm**2 / u.AA
        flux = data["FLUX"][mask] * u.erg / u.s / u.cm**2 / u.AA
        spec_data = [wavs, flux, cont]

    else:
        wavs = data["WAVELENGTH"][mask] * u.AA
        flux = data["FLUX"][mask] * u.erg / u.s / u.cm**2 / u.AA
        stat_err = data["STATERROR"][mask] * u.erg / u.s / u.cm**2 / u.AA
        syst_err = data["SYSERROR"] * u.erg / u.s / u.cm**2 / u.AA
        fwhm = data["FWHM"][mask] * u.AA
        data_quality = data["DATAQUAL"][mask]
        exp_time = data["TOTEXP"][mask] * u.s
        spec_data = [wavs, flux, stat_err, syst_err, fwhm, data_quality, exp_time]

    if debug:
        plt.rcParams["figure.figsize"] = (12, 4)
        plt.plot(spec_data[0], spec_data[1], color="k")
        plt.xlim(np.min(spec_data[0].value), np.max(spec_data[0].value))
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Wavelength [Angstroms]")
        plt.ylabel("Flux [flam]")
        plt.show()

    return spec_data


def generate_flux_conversion(
    w_measured: np.ndarray,
    w_model: np.ndarray,
    f_measured: np.ndarray,
    f_model: np.ndarray,
    err: np.ndarray,
    sigma_clip: float = 50.0,
    order: int = 7,
    max_iter: int = 50,
    debug: bool = False,
) -> np.poly1d:
    """
    Generates a numpy polynomial that predicts the physical flux [flam]
    / CCD count as a function of wavelength [Angstroms].

    Parameters:
    -----------
    w_measured :: np.ndarray
        A 1D array of wavelengths for your CCD spectrum (in Angstroms).
    w_model :: np.ndarray
        A 1D array of wavelengths for your known spectrum (in Angstroms).
    f_measured :: np.ndarray
        A 1D array of flux for your CCD spectrum.
    f_model :: np.ndarray
        A 1D array of flux for your known spectrum.
    err :: np.ndarray
        A 1D array of errors for you CCD flux.
    sigma_clip :: float
        The max number of standard deviations a point is allowed to be
        from the calibration model. Outlier points are removed from
        future fits.
    order :: int
        Polynomial order of the flux conversion.
    max_iter :: int
        Maximum number of fits to perform before manually stopping.
    debug :: bool
        Plots the final fit against the user-provided data.

    Returns:
    --------
    p_flux_conversion :: np.poly1d
        An n-th order polynomial that takes wavelength [Angstroms] as
        an argument. The returned value converts CCD counts into
        physical units [flam], meaning it is in units of flam/count.
    """

    # Ensures that none of the arrays have associated units
    if isinstance(w_measured, Quantity):
        w_measured = (w_measured.to(u.AA)).value
    if isinstance(w_model, Quantity):
        w_model = (w_model.to(u.AA)).value
    if isinstance(f_measured, Quantity):
        f_measured = f_measured.value
    if isinstance(f_model, Quantity):
        f_model = f_model.value

    # Ensures all arrays are 1D
    w_measured = w_measured.flatten()
    w_model = w_model.flatten()
    f_measured = f_measured.flatten()
    f_model = f_model.flatten()
    err = err.flatten()

    # Initializes arrays for iterative fitting
    weights = 1 / err
    mask = np.array([True for _ in f_measured])
    old_mask = np.array([False for _ in f_measured])

    step = 0

    # Runs until the "best" fit is reached
    while step <= max_iter and not np.array_equal(mask, old_mask):

        # Fits for the conversion between both spectra
        interp_flux = np.interp(w_measured, w_model, f_model)
        ratio = interp_flux / f_measured
        coeffs = np.polyfit(w_measured[mask], ratio[mask], order, w=weights[mask])
        p_flux_conversion = np.poly1d(coeffs)

        # Updates fit weights
        weights = f_measured / (err * p_flux_conversion(w_measured))

        # Updates mask arrays
        old_mask = mask.copy()
        mask = mask & (
            ((p_flux_conversion(w_measured) - ratio) * weights) ** 2 < sigma_clip
        )

        step += 1

    if debug:
        plt.rcParams["figure.figsize"] = (12, 4)
        plt.scatter(
            w_measured,
            f_measured * p_flux_conversion(w_measured),
            s=1,
            color="k",
            label="Flux-Calibrated",
        )
        plt.plot(w_model, f_model, color="red", lw=1, label="Model")
        plt.xlim(min(w_measured), max(w_measured))
        plt.xlabel("Wavelength [Angstroms]")
        plt.ylabel("Flux [flam]")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.show()

    return p_flux_conversion
