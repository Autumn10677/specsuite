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

    def test_estimate_extinction_coefficients(self):

        # Flux and error arrays must have the same shape
        for _ in range(5):
            with self.assertRaises(AssertionError):
                extraction.estimate_extinction_coefficients(
                    airmass=np.random.rand(3),
                    flux=np.random.rand(3, 3),
                    error=np.random.rand(4, 4),
                )

        # Airmass array must have a compatible shape with flux array
        for _ in range(5):
            with self.assertRaises(AssertionError):
                extraction.estimate_extinction_coefficients(
                    airmass=np.random.rand(4),
                    flux=np.random.rand(3, 3),
                    error=np.random.rand(3, 3),
                )

        # Max iterations must be positive
        with self.assertRaises(AssertionError):
            extraction.estimate_extinction_coefficients(
                airmass=np.random.rand(3),
                flux=np.random.rand(3, 3),
                error=np.random.rand(3, 3),
                max_iterations=-1,
            )

        # Max iterations cannot be 0
        with self.assertRaises(AssertionError):
            extraction.estimate_extinction_coefficients(
                airmass=np.random.rand(3),
                flux=np.random.rand(3, 3),
                error=np.random.rand(3, 3),
                max_iterations=0,
            )

        # Clip threshold must be positive
        with self.assertRaises(AssertionError):
            extraction.estimate_extinction_coefficients(
                airmass=np.random.rand(3),
                flux=np.random.rand(3, 3),
                error=np.random.rand(3, 3),
                clip_threshold=-1.0,
            )

    def test_fit_extinction_model(self):

        bad_p0 = {
            "rs_tau0": 0.1,
            "o2_abundance": 0.2,
            "humidity": 0.3,
            "w_offset": 0.4,
        }
        bad_bounds = {
            "rs_tau0": (0.0, 1.0),
            "o2_abundance": (0.0, 1.0),
            "humidity": (0.0, 1.0),
            "w_offset": (-1.0, 1.0),
        }

        # Ensures that missing keys in p0 and bounds will raise an error
        with self.assertRaises(AssertionError):
            extraction.fit_extinction_model(
                wavelengths=np.linspace(700, 800, 100),
                taus=np.random.rand(100),
                taus_error=np.random.rand(100) * 0.1,
                p0=bad_p0,
            )
        with self.assertRaises(AssertionError):
            extraction.fit_extinction_model(
                wavelengths=np.linspace(700, 800, 100),
                taus=np.random.rand(100),
                taus_error=np.random.rand(100) * 0.1,
                bounds=bad_bounds,
            )

    def test_fit_achromatic_bumps(self):

        # Ensures that flux and error arrays must have the same shape
        with self.assertRaises(AssertionError):
            extraction.fit_achromatic_bumps(
                times=np.random.rand(10),
                flux=np.random.rand(20, 100),
                error=np.random.rand(10, 100) * 0.1,
                airmasses=np.random.rand(10),
                throughput_model=None,
                max_iterations=5,
                outlier_threshold=3,
            )

        # Ensures that time and airmass arrays must have compatible shapes
        with self.assertRaises(AssertionError):
            extraction.fit_achromatic_bumps(
                times=np.random.rand(20),
                flux=np.random.rand(10, 100),
                error=np.random.rand(10, 100) * 0.1,
                airmasses=np.random.rand(10),
                throughput_model=None,
                max_iterations=5,
                outlier_threshold=3,
            )

        # Ensures that times must have a compatible shape with flux array
        with self.assertRaises(AssertionError):
            extraction.fit_achromatic_bumps(
                times=np.random.rand(20),
                flux=np.random.rand(10, 100),
                error=np.random.rand(10, 100) * 0.1,
                airmasses=np.random.rand(20),
                throughput_model=None,
                max_iterations=5,
                outlier_threshold=3,
            )


if __name__ == "__main__":
    unittest.main()
