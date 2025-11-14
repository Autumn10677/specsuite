import sys
import unittest
import warnings
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

sys.path.append("specsuite/")

import loading, throughput  # noqa

CAL_PATH = "data/KOSMOS/calibrations"
DATA_PATH = "data/KOSMOS/target"

class TestLoadingFunctions(unittest.TestCase):

    def test_load_STIS_spectra(self):

        valid_names = ["10 LAC", "GD153", "NGC2506-G31"]

        # Checks that known function behavior works
        for name in valid_names:

            spec_data = throughput.load_STIS_spectra(
                name,
                wavelength_bounds=(300*u.nm, 900*u.nm),
            )
            wavelength, flux, continuum = spec_data

            self.assertTrue(
                (len(wavelength) == len(flux))
                and (len(wavelength) == len(continuum))
            )

        # Checks expected user errors / handling
        with self.assertRaises(AssertionError):
            throughput.load_STIS_spectra("fake name")
        with self.assertRaises(AssertionError):
            throughput.load_STIS_spectra("GD153",filetype="bad model type")
        with self.assertRaises(AssertionError):
            throughput.load_STIS_spectra("GD153", wavelength_bounds = "Test")

if __name__ == "__main__":
    unittest.main()
