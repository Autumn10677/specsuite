import unittest

import specsuite.utils as utils
import specsuite.loading as loading

CAL_PATH = "data/KOSMOS/calibrations"
DATA_PATH = "data/KOSMOS/target"

bias = loading.average_matching_files(CAL_PATH, "bias")
flat = loading.average_matching_files(CAL_PATH, "flat") - bias
science = loading.collect_images_array(DATA_PATH, "toi3884") - bias


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


if __name__ == "__main__":
    unittest.main()
