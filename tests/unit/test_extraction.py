import unittest
import numpy as np

import specsuite.loading as loading
import specsuite.warping as warping
import specsuite.extraction as extraction

CAL_PATH = "data/KOSMOS/calibrations"
DATA_PATH = "data/KOSMOS/target"
REGION = (700, 800)

bias = loading.average_matching_files(CAL_PATH, "bias", crop_bds=REGION)
arc = loading.average_matching_files(CAL_PATH, "neon", crop_bds=REGION) - bias
science = loading.collect_images_array(DATA_PATH, "toi3884", crop_bds=REGION) - bias

locs, _ = warping.find_cal_lines(arc, std_variation=200)
warp_model = warping.generate_warp_model(arc, locs)
backgrounds = warping.extract_background(science, warp_model)
science -= backgrounds


class TestExtractionFunctions(unittest.TestCase):

    def test_boxcar_extraction(self):

        flux, error = extraction.boxcar_extraction(
            images=science,
            backgrounds=backgrounds,
        )

        # Ensures that both arrays are 2D arrays of the same shape
        self.assertTrue(flux.shape == error.shape)
        self.assertTrue(len(flux.shape) == 2)

    def test_spatial_profile(self):

        P_moffat = extraction.generate_spatial_profile(science[0], profile="moffat")
        P_gauss = extraction.generate_spatial_profile(science[0], profile="gaussian")

        # Ensures that both profiles produce a single 2D image
        self.assertTrue(len(P_moffat.shape) == 2)
        self.assertTrue(len(P_gauss.shape) == 2)

        # Ensures that profile shape matches exposure shape
        self.assertTrue(P_moffat.shape == science[0].shape)
        self.assertTrue(P_gauss.shape == science[0].shape)

        # Makes sure that invalid profiles will terminate code
        with self.assertRaises(AssertionError):
            extraction.generate_spatial_profile(science[0], profile="bad profile")

    def test_horne_extraction(self):

        # Valid call for single, constant value of RN
        flux, error = extraction.horne_extraction(
            images=science[:2], backgrounds=backgrounds[:2], RN=6.0, profile="moffat"
        )

        # Ensures that both arrays are 2D arrays of the same shape
        self.assertTrue(flux.shape == error.shape)
        self.assertTrue(len(flux.shape) == 2)

        # Valid call using 2D array for RN
        RN_array = np.zeros(shape=science[0].shape)
        flux, error = extraction.horne_extraction(
            images=science[:2],
            backgrounds=backgrounds[:2],
            RN=RN_array,
            profile="moffat",
        )

        # A separate sanity check for when RN is 2D array
        self.assertTrue(flux.shape == error.shape)
        self.assertTrue(len(flux.shape) == 2)


if __name__ == "__main__":
    unittest.main()
