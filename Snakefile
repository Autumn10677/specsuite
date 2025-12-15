import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import specsuite as ss

import matplotlib
matplotlib.use('Agg')

try:
    os.mkdir("outputs")
except FileExistsError:
    pass

sys.path.append("specsuite/")  # noqa

CAL_PATH = "data/KOSMOS/calibrations"
DATA_PATH = "data/KOSMOS/target"
DATA_REGION = (700, 800)
VMIN, VMAX = (1e2, 1e4)

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
        "outputs/flux.npy",
        "outputs/flux_errs.npy",

rule basic_calibrations:
    output:
        "outputs/median_bias.npy",
        "outputs/median_flat.npy",
        "outputs/median_arc.npy",
        "outputs/data_array.npy",
    run:
        bias = ss.average_matching_files(path = CAL_PATH, tag = "bias", crop_bds=DATA_REGION)
        flat = ss.average_matching_files(path = CAL_PATH, tag = "flat", crop_bds=DATA_REGION) - bias
        arc = ss.average_matching_files(path = CAL_PATH, tag = "neon", crop_bds=DATA_REGION) - bias
        data = ss.collect_images_array(path = DATA_PATH, tag="toi3884", crop_bds=DATA_REGION) - bias

        ss.plot_image(bias, title="Bias Exposure", savedir="outputs/bias.png")
        ss.plot_image(flat, title="Flat Exposure", savedir="outputs/flat.png")
        ss.plot_image(arc, title="Arclamp Exposure", savedir="outputs/arclamp.png")
        ss.plot_image(data[0], norm='log', vmin=VMIN, vmax=VMAX, title="Raw Data", savedir="outputs/raw_data.png")

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

        locs, _ = ss.find_cal_lines(arc, std_variation=50)
        warp_model = ss.generate_warp_model(arc, locs)
        background = ss.extract_background(data, warp_model, mask_region=(40, 60))

        ss.plot_image(background[0], norm='log', vmin=VMIN, vmax=VMAX, title="Extracted Background", savedir="outputs/background.png")
        ss.plot_image(data[0] - background[0], norm='log', vmin=VMIN, vmax=VMAX, title="Background-Corrected Data", savedir="outputs/background_corrected_data.png")
        np.save("outputs/background_array.npy", background)

rule flux_extraction:
    input:
        "outputs/background_array.npy",
    output:
        "outputs/flux.npy",
        "outputs/flux_errs.npy",
    run:
        data = np.load("outputs/data_array.npy")
        backgrounds = np.load("outputs/background_array.npy")

        data -= backgrounds

        flux, errs = ss.boxcar_extraction(data, backgrounds, RN=6.0)

        np.save("outputs/flux.npy", flux)
        np.save("outputs/flux_errs.npy", errs)