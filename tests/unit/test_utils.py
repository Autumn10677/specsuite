import unittest
import numpy as np

import specsuite.utils as utils
import specsuite.loading as loading

CAL_PATH = "data/KOSMOS/calibrations"
DATA_PATH = "data/KOSMOS/target"

bias = loading.average_matching_files(path=CAL_PATH, tag="bias")
flat = (
    loading.average_matching_files(
        path=CAL_PATH,
        tag="flat",
        ignore=["flat.0029.fits", "flat.0030.fits"],
    )
    - bias
)
science = loading.collect_images_array(path=DATA_PATH, tag="toi3884") - bias


class TestUtilFunctions(unittest.TestCase):

    def test_plot_image(self):

        # Ensures that plotting will not work for invalid image shapes
        with self.assertWarns(UserWarning):
            utils.plot_image([], norm="log")
        with self.assertWarns(UserWarning):
            utils.plot_image([1, 2, 3, 4], norm="log")
        with self.assertWarns(UserWarning):
            utils.plot_image([[[1, 2], [3, 4], [5, 6]]], norm="log")

    def test_gaussian_profile(self):
        self.assertEqual(
            utils._gaussian(
                x=["a", "b", "c"],
                A=1,
                mu=1,
                sigma=1,
            ),
            None,
        )

        self.assertEqual(
            utils._gaussian(
                x=[1, 2, 3, 4],
                A="a",
                mu=10,
                sigma=10,
            ),
            None,
        )

        self.assertEqual(
            len(
                utils._gaussian(
                    x=[1, 2, 3, 4],
                    A=0,
                    mu=10,
                    sigma=10,
                ),
            ),
            4,
        )

        self.assertEqual(
            len(
                utils._gaussian(
                    x=[],
                    A=0,
                    mu=10,
                    sigma=10,
                ),
            ),
            0,
        )

    def test_moffat_profile(self):
        self.assertEqual(
            utils._moffat(
                x=["a", "b", "c"],
                A=1,
                mu=1,
                gamma=1,
            ),
            None,
        )

        self.assertEqual(
            utils._moffat(
                x=[1, 2, 3, 4],
                A="a",
                mu=10,
                gamma=10,
            ),
            None,
        )

        self.assertEqual(
            len(
                utils._moffat(
                    x=[1, 2, 3, 4],
                    A=0,
                    mu=10,
                    gamma=10,
                ),
            ),
            4,
        )

        self.assertEqual(
            len(
                utils._moffat(
                    x=[],
                    A=0,
                    mu=10,
                    gamma=10,
                ),
            ),
            0,
        )

    def test_rebin_image_columns(self):
        binned_image = utils.rebin_image_columns(bias, bin=4)

        # Checks that binning changes axis=1 by the expected amount
        self.assertTrue(bias.shape[0] == binned_image.shape[0])
        self.assertTrue(bias.shape[1] == binned_image.shape[1] * 4)

        # Makes sure that passing a float will throw an error
        with self.assertRaises(AssertionError):
            utils.rebin_image_columns(bias, bin=1.2)

    def test_flatfield_correction(self):

        # Ensures that the output retains the original shape
        valid_output = utils.flatfield_correction(science, flat)
        self.assertTrue(valid_output.shape == science.shape)

        # Ensures that invalid shapes are caught before running correction
        with self.assertRaises(AssertionError):
            utils.flatfield_correction(science, [[], []])

    def test_convolve_to_resolution(self):

        # Ensures that x and y must have the same length
        with self.assertRaises(AssertionError):
            utils.convolve_to_resolution(
                x=[1, 2, 3, 4, 5],
                y=[1, 2, 3, 4, 5, 6],
                R=1000,
            )

        # Spectral resolution cannot be zero
        with self.assertRaises(AssertionError):
            utils.convolve_to_resolution(
                x=[1, 2, 3, 4, 5, 6],
                y=[1, 2, 3, 4, 5, 6],
                R=0,
            )

        # Spectral resolution also cannot be negative
        with self.assertRaises(AssertionError):
            utils.convolve_to_resolution(
                x=[1, 2, 3, 4, 5, 6],
                y=[1, 2, 3, 4, 5, 6],
                R=-10,
            )

        # Cannot have NaN values in your wavelengths
        with self.assertRaises(AssertionError):
            utils.convolve_to_resolution(
                x=[1, 2, 3, 4, 5, np.nan],
                y=[1, 2, 3, 4, 5, 6],
                R=1000,
            )

        # Also cannot have NaN values in your fluxes
        with self.assertRaises(AssertionError):
            utils.convolve_to_resolution(
                x=[1, 2, 3, 4, 5, 6],
                y=[1, 2, 3, 4, 5, np.nan],
                R=1000,
            )

        # Runs 10 random, valid calls to test expected behavior
        for _ in range(10):
            x = np.sort(np.random.uniform(5000, 6000, size=1000))
            y = np.random.normal(loc=1.0, scale=0.1, size=1000)
            R = np.random.uniform(500, 20000)

            convolved_y = utils.convolve_to_resolution(x, y, R)

            # Output should have the same shape as the input
            self.assertTrue(convolved_y.shape == y.shape)

    def test_peak_phase_shift(self):

        x = np.arange(0, 100)
        easy_offset = 5.0
        hard_offset = 1.3

        # Generates three identical arrays with a constant offset
        profile1 = utils._gaussian(
            x=x,
            A=1.0,
            mu=50,
            sigma=10,
        )
        profile2 = utils._gaussian(
            x=x,
            A=1.0,
            mu=50 + easy_offset,
            sigma=10,
        )
        profile3 = utils._gaussian(
            x=x,
            A=1.0,
            mu=50 + hard_offset,
            sigma=10,
        )

        # Function inputs must be FFTs, not raw data
        profile1_fft = np.fft.fft(profile1)
        profile2_fft = np.fft.fft(profile2)
        profile3_fft = np.fft.fft(profile3)

        # This should be approximately correct
        estimated_offset = utils.peak_phase_shift(
            profile1_fft,
            profile2_fft,
        )
        self.assertAlmostEqual(easy_offset, estimated_offset, 3)

        # This should be slightly less correct, but still reasonable
        estimated_offset = utils.peak_phase_shift(
            profile1_fft,
            profile3_fft,
        )
        self.assertAlmostEqual(hard_offset, estimated_offset, 1)

    def test_estimate_exposure_offsets(self):

        flux = np.ones((20, 20))

        valid_flux_model = np.ones(20)
        invalid_flux_model = valid_flux_model[:-1]

        # Ensures that invalid modes raise an error
        with self.assertRaises(AssertionError):
            utils.estimate_exposure_offsets(flux=flux, mode="bad mode")

        # A float 'poly_order' would cause an error during fitting
        with self.assertRaises(AssertionError):
            utils.estimate_exposure_offsets(
                flux=flux,
                mode="fit",
                N_divisions=3,
                poly_order=1.1,
            )

        # Ensures 'N_divisions' >= 'poly_order'
        with self.assertRaises(AssertionError):
            utils.estimate_exposure_offsets(
                flux=flux,
                mode="fit",
                N_divisions=3,
                poly_order=4,
            )

        # 'flux' and 'model_flux' must have compatible shapes
        with self.assertRaises(AssertionError):
            utils.estimate_exposure_offsets(
                flux=flux,
                model_flux=invalid_flux_model,
            )

        # Performs a simple, valid function call
        _ = utils.estimate_exposure_offsets(
            flux=flux,
            model_flux=valid_flux_model,
        )

    def test_perform_cdf_interpolation(self):

        # Defines some valid arrays for interpolation
        x_initial = np.arange(0, 100)
        x_desired = x_initial + 5.32
        flux = utils._gaussian(
            x=x_initial,
            A=1.0,
            mu=50,
            sigma=10,
        )
        error = np.sqrt(flux)

        interp_flux, _ = utils.perform_cdf_interpolation(
            flux=flux,
            error=error,
            current_pixels=x_initial,
            target_pixels=x_desired,
        )

        # Ensures the original flux is relatively conserved
        self.assertAlmostEqual(np.sum(flux), np.nansum(interp_flux), 5)

        # Ensures that identical grid produces nearly identical flux / error
        interp_flux, interp_error = utils.perform_cdf_interpolation(
            flux=flux,
            error=error,
            current_pixels=x_initial,
            target_pixels=x_initial,
        )
        self.assertTrue(
            np.nanmax(np.abs(flux - interp_flux)) < 1e-12,
        )
        self.assertTrue(
            np.nanmax(np.abs(error - interp_error)) < 1e-12,
        )

        # Flux array must be 1-dimensional for interpolation
        with self.assertRaises(AssertionError):
            utils.perform_cdf_interpolation(
                flux=np.ones((2, 2)),
                error=error,
                current_pixels=x_initial,
                target_pixels=x_initial,
            )

        # Error array must have the same shape as fluxes
        with self.assertRaises(AssertionError):
            utils.perform_cdf_interpolation(
                flux=flux,
                error=error[:-1],
                current_pixels=x_initial,
                target_pixels=x_initial,
            )

        # Initial pixel positions must have same shape as fluxes
        with self.assertRaises(AssertionError):
            utils.perform_cdf_interpolation(
                flux=flux,
                error=error,
                current_pixels=x_initial[:-1],
                target_pixels=x_initial,
            )

        # Current pixels must be sorted
        with self.assertRaises(AssertionError):
            utils.perform_cdf_interpolation(
                flux=np.ones(3),
                error=np.ones(3),
                current_pixels=np.array([0, 2, 1]),
                target_pixels=np.array([0, 1, 2]),
            )

        # Target pixels must be sorted
        with self.assertRaises(AssertionError):
            utils.perform_cdf_interpolation(
                flux=np.ones(3),
                error=np.ones(3),
                current_pixels=np.array([0, 1, 2]),
                target_pixels=np.array([0, 2, 1]),
            )


if __name__ == "__main__":
    unittest.main()
