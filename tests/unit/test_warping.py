import unittest
import numpy as np

import specsuite.loading as loading
import specsuite.warping as warping
import specsuite.utils as utils

CAL_PATH = "data/KOSMOS/calibrations"
DATA_PATH = "data/KOSMOS/target"

trace_region = (150, 300)

# Gathers and calibrates images used across all tests
bias = loading.average_matching_files(CAL_PATH, "bias", crop_bds=trace_region)
flat = loading.average_matching_files(CAL_PATH, "flat", crop_bds=trace_region) - bias
arc = loading.average_matching_files(CAL_PATH, "neon", crop_bds=trace_region) - bias
data = loading.collect_images_array(DATA_PATH, "", crop_bds=trace_region) - bias
arc = utils.flatfield_correction(arc, flat)


class TestWarpingFunctions(unittest.TestCase):

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
            "No pixels were found above the provided threshold (10000.0)",
        )

    def test_combine_within_tolerance(self):

        example_list = [1, 2, 5, 6, 10]

        self.assertTrue(
            np.array_equal(
                warping.combine_within_tolerance(example_list, 1),
                np.array([1.5, 5.5, 10.0]),
            )
        )

        self.assertTrue(
            np.array_equal(
                warping.combine_within_tolerance(example_list, 0.999),
                np.array([1.0, 2.0, 5.0, 6.0, 10.0]),
            )
        )

        self.assertTrue(
            np.array_equal(warping.combine_within_tolerance([], 1.0), np.array([]))
        )

        self.assertTrue(
            np.array_equal(
                warping.combine_within_tolerance([1, 2, 3, 4], None), np.array([])
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

    def test_generate_warp_model(self):

        # Sanity checks for a valid example call
        locs, _ = warping.find_cal_lines(arc)
        warp_models = warping.generate_warp_model(arc, locs)
        self.assertTrue(len(warp_models) > 0)
        self.assertIsInstance(warp_models, list)
        self.assertIsInstance(warp_models[0], np.poly1d)

        # Checks known error handling for expected user mistakes
        with self.assertWarns(UserWarning):

            warp_models = warping.generate_warp_model(arc, [])
            self.assertTrue(warp_models is None)

            warp_models = warping.generate_warp_model(
                image=arc,
                guess=locs,
                tolerance=None,
            )

            warp_models = warping.generate_warp_model(
                image=arc,
                guess=locs,
                line_order=None,
            )

            warp_models = warping.generate_warp_model(
                image=arc,
                guess=locs,
                warp_order=None,
            )

    def test_dewarp_image(self):

        # Sanity checks for a valid example call
        locs, _ = warping.find_cal_lines(arc)
        warp_models = warping.generate_warp_model(arc, locs)
        dewarped_image = warping.dewarp_image(arc, warp_models, progress=True)
        self.assertTrue(dewarped_image.shape == arc.shape)
        self.assertIsInstance(dewarped_image, np.ndarray)

        # If multiple images are provided, spit back out the original array
        with self.assertWarns(UserWarning):
            test_array = warping.dewarp_image(data, warp_models, progress=True)
            self.assertTrue(np.array_equal(test_array, data))

    def test_extract_background(self):

        # Sanity check for simple, valid call of background extraction
        locs, _ = warping.find_cal_lines(arc)
        warp_models = warping.generate_warp_model(arc, locs)
        backgrounds = warping.extract_background(
            data,
            warp_models,
            mask_region=(60, 100),
            progress=True,
        )
        self.assertTrue(backgrounds.shape == data.shape)
        self.assertIsInstance(backgrounds, np.ndarray)

        # Sanity check for complete, valid call of background extraction
        bgs, super_effpix, super_spectra, effpix_map = warping.extract_background(
            data,
            warp_models,
            mask_region=(60, 100),
            progress=True,
            return_spectrum=True,
        )
        self.assertTrue(bgs.shape == data.shape)
        self.assertIsInstance(bgs, np.ndarray)
        self.assertTrue(super_spectra.shape == (len(data), len(super_effpix)))
        self.assertTrue(effpix_map.shape == data[0].shape)


if __name__ == "__main__":
    unittest.main()
