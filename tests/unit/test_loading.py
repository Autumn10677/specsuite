import unittest
import numpy as np

import specsuite.loading as loading

CAL_PATH = "data/KOSMOS/calibrations"
DATA_PATH = "data/KOSMOS/target"


class TestLoadingFunctions(unittest.TestCase):

    def test_filter_files(self):

        filenames = [
            "toi3884_0001.fits",
            "neptune_01.fits",
            "toi3884_0002.fits",
            "toi3884_0003.fits",
            "neptune_02.png",
            "neptune_03.html",
            "neptune_04.fits",
            "toi3884_0004.csv",
        ]

        basic_filter = loading.filter_files(
            files=filenames,
            tag="",
            ignore=[],
        )

        toi3884_filter = loading.filter_files(
            files=filenames,
            tag="toi3884",
            ignore=["toi3884_0003.fits"],
        )

        neptune_filter = loading.filter_files(
            files=filenames,
            tag="neptune",
            ignore=[],
        )

        bad_filter = loading.filter_files(
            files=filenames,
            tag="bad tag",
            ignore=[],
        )

        self.assertEqual(
            basic_filter,
            [
                "neptune_01.fits",
                "neptune_04.fits",
                "toi3884_0001.fits",
                "toi3884_0002.fits",
                "toi3884_0003.fits",
            ],
        )
        self.assertEqual(toi3884_filter, ["toi3884_0001.fits", "toi3884_0002.fits"])
        self.assertEqual(neptune_filter, ["neptune_01.fits", "neptune_04.fits"])
        self.assertEqual(bad_filter, [])

    def test_collect_images_array(self):

        # Ensures that valid call returns a numpy array (instead of 'None')
        valid_images = loading.collect_images_array(path=DATA_PATH, tag="toi3884")
        self.assertTrue(len(valid_images) > 0)
        self.assertIsInstance(valid_images, np.ndarray)

        # Since 'wrong tag' is not in any filename, no files should be loaded
        self.assertTrue(
            loading.collect_images_array(path=CAL_PATH, tag="wrong tag") is None
        )

        # FIXME: In the future, bad instruments should default to 'default' loading
        self.assertTrue(
            loading.collect_images_array(
                path=CAL_PATH,
                tag="bias",
                instrument="fake instrument",
            )
            is None
        )

    def test_average_matching_files(self):

        # Ensures that valid call returns a numpy array (instead of 'None')
        valid_images = loading.average_matching_files(
            path=DATA_PATH,
            tag="toi3884",
        )
        self.assertTrue(len(valid_images) > 0)
        self.assertIsInstance(valid_images, np.ndarray)

        self.assertTrue(
            loading.average_matching_files(path=CAL_PATH, tag="wrong tag") is None
        )

        self.assertTrue(
            loading.average_matching_files(
                path=CAL_PATH,
                tag="bias",
                instrument="fake instrument",
            )
            is None
        )


if __name__ == "__main__":
    unittest.main()
