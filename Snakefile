import os
import sys
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.use('Agg')

try:
    os.mkdir("outputs")
except FileExistsError:
    pass

sys.path.append("specsuite/")  # noqa

import loading
import utils
import warping

CAL_PATH = "data/KOSMOS/calibrations"
DATA_PATH = "data/KOSMOS/target"
DATA_REGION = (700, 850)

PLOTTING_KWARGS = {
    "cmap": "inferno",
    "origin": "lower",
    "interpolation": "none",
    "norm": "log",
    "aspect": "auto",
}

rule all:
    input:
        "outputs/median_bias.npy",
        "outputs/median_flat.npy",
        "outputs/median_arc.npy",
        "outputs/data_array.npy",
        "outputs/background_array.npy",
        #"outputs/flux.npy",

rule basic_calibrations:
    output:
        "outputs/median_bias.npy",
        "outputs/median_flat.npy",
        "outputs/median_arc.npy",
        "outputs/data_array.npy",
    run:
        print(config["testarg"])

        bias = loading.average_matching_files(path = CAL_PATH, tag = "bias", crop_bds=DATA_REGION)
        flat = loading.average_matching_files(path = CAL_PATH, tag = "flat", crop_bds=DATA_REGION) - bias
        arc = loading.average_matching_files(path = CAL_PATH, tag = "neon", crop_bds=DATA_REGION) - bias
        data = loading.collect_images_array(path = DATA_PATH, tag="toi3884", crop_bds=DATA_REGION) - bias

        data = utils.flatfield_correction(data, arc)
        data = utils.flatfield_correction(data, flat)

        np.save("outputs/median_bias.npy", bias)
        np.save("outputs/median_flat.npy", flat)
        np.save("outputs/median_arc.npy", arc)
        np.save("outputs/data_array.npy", data)

rule background_extraction:
    input:
        "outputs/median_bias.npy",
        "outputs/median_flat.npy",
        "outputs/median_arc.npy",
        "outputs/data_array.npy",
    output:
        "outputs/background_array.npy",
    run:
        arc = np.load("outputs/median_arc.npy")
        data = np.load("outputs/data_array.npy")

        locs, _ = warping.find_cal_lines(arc)
        warp_model = warping.generate_warp_model(arc, locs)
        backgrounds = warping.extract_background(data, warp_model, mask_region=(40, 60))

        np.save("outputs/background_array.npy", backgrounds)

# rule flux_extraction:
#     input:
#         "outputs/background_array.npy",
#     output:
#         "outputs/flux.npy",
#         "outputs/flux_errs.npy",
#     run:
#         data = np.load("outputs/data_array.npy")
#         backgrounds = np.load("outputs/background_array.npy")

#         data -= backgrounds

#         flux, errs = extraction.horne_extraction(data, backgrounds)