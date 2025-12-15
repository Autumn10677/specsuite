import sys
import unittest
import warnings
import numpy as np

import specsuite.utils as utils
import specsuite.loading as loading

# sys.path.append("specsuite/")

# import utils
# import loading

CAL_PATH = "data/KOSMOS/calibrations"

class TestUtilFunctions(unittest.TestCase):

    def test_plot_image(self):

        # Ensures that plotting will not work for invalid image shapes
        with self.assertWarns(UserWarning):
            utils.plot_image([], norm='log')
        with self.assertWarns(UserWarning):
            utils.plot_image([1, 2, 3, 4], norm='log')
        with self.assertWarns(UserWarning):
            utils.plot_image([[[1, 2], [3, 4], [5, 6]]], norm='log')

    def test_plot_spectra(self):
        pass

    def test_gaussian_profile(self):
        self.assertEqual(
            utils._gaussian(
                x=["a", "b", "c"], A=1, mu=1, sigma=1,
            ),
            None
        )

        self.assertEqual(
            utils._gaussian(
                x=[1, 2, 3, 4], A="a", mu=10, sigma=10,
            ),
            None
        )

        self.assertEqual(
            len(utils._gaussian(
                    x=[1, 2, 3, 4], A=0, mu=10, sigma=10,
                ),
            ),
            4
        )

        self.assertEqual(
            len(utils._gaussian(
                    x=[], A=0, mu=10, sigma=10,
                ),
            ),
            0
        )

    def test_moffat_profile(self):
        self.assertEqual(
            utils._moffat(
                x=["a", "b", "c"], A=1, mu=1, gamma=1,
            ),
            None
        )

        self.assertEqual(
            utils._moffat(
                x=[1, 2, 3, 4], A="a", mu=10, gamma=10,
            ),
            None
        )

        self.assertEqual(
            len(utils._moffat(
                    x=[1, 2, 3, 4], A=0, mu=10, gamma=10,
                ),
            ),
            4
        )

        self.assertEqual(
            len(utils._moffat(
                    x=[], A=0, mu=10, gamma=10,
                ),
            ),
            0
        )

    def test_rebin_image_columns(self):
        pass

    def test_flatfield_correction(self):
        pass

if __name__ == "__main__":
    unittest.main()
