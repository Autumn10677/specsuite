import sys
import unittest
import warnings
import numpy as np

sys.path.append("specsuite/")

import loading  # noqa
import warping
import utils

CAL_PATH = "data/KOSMOS/calibrations"
DATA_PATH = "data/KOSMOS/target"

trace_region = (150, 300)

# Gathers images that will be used for all tests
bias = loading.average_matching_files(CAL_PATH, "bias", crop_bds=trace_region)
flat = loading.average_matching_files(CAL_PATH, "flat", crop_bds=trace_region) - bias
arc = loading.average_matching_files(CAL_PATH, "neon", crop_bds=trace_region) - bias
data = loading.collect_images_array(DATA_PATH, "", crop_bds=trace_region) - bias

arc = utils.flatfield_correction(arc, flat)

#utils.plot_image(data[0], norm='log')

class TestLoadingFunctions(unittest.TestCase):

    def test_find_cal_lines(self):

        # Sanity checks for valid function call
        locs, ints = warping.find_cal_lines(arc, std_variation=100)
        self.assertTrue(len(locs) > 0)
        self.assertEqual(len(locs), len(ints))
        self.assertIsInstance(locs, np.ndarray)
        self.assertIsInstance(ints, np.ndarray)

        # Ensures that using too high of a threshold throws the proper error
        with self.assertRaises(ZeroDivisionError) as cm:
            _, _ = warping.find_cal_lines(arc, std_variation=1e4, debug=True)
        self.assertEqual(
            str(cm.exception),
            "No pixels were found above the provided threshold (10000.0)"
        )

    def test_combine_within_tolerance(self):

        example_list = [1, 2, 5, 6, 10]

        self.assertTrue(
            np.array_equal(
                warping.combine_within_tolerance(example_list, 1),
                np.array([1.5, 5.5, 10.])
            )
        )

        self.assertTrue(
            np.array_equal(
                warping.combine_within_tolerance(example_list, 0.999),
                np.array([1., 2., 5., 6., 10.])
            )
        )

        self.assertTrue(
            np.array_equal(
                warping.combine_within_tolerance([], 1.0),
                np.array([])
            )
        )

        self.assertTrue(
            np.array_equal(
                warping.combine_within_tolerance([1, 2, 3, 4], None),
                np.array([])
            )
        )

        self.assertTrue(
            np.array_equal(
                warping.combine_within_tolerance(["a", "b", "c"], 1.0),
                np.array(["a", "b", "c"]),
            )
        )

        self.assertTrue(
            np.array_equal(
                warping.combine_within_tolerance([1, 2, 10], -1),
                warping.combine_within_tolerance([1, 2, 10], 1),
            )
        )

if __name__ == "__main__":
    unittest.main()
