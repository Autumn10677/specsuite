# Welcome to specsuite!
For help with getting ```specsuite``` running on your own data, please check out the <a href="https://www.autumnstephens.net/specsuite" target="_blank"><strong>documentation page</strong></a>!

## Introduction
Although other spectroscopic reduction tools exist, they are often designed for a small subset of instruments, have hard-to-read documentation, or are difficult to debug. ```specsuite``` was designed to address all three of these concerns, providing a set of robust, generalized, and user-friendly reduction tools! As of writing, this reduction pipeline has been tested this reduction pipeline against data from...

- Gemini North (GMOS-N)
- Apache Point Observatory (KOSMOS)
- Sommers-Bausch Observatory (SBO)

...but we are constantly testing on data from other telescopes!

Another advantage of ```specsuite``` is its modularity. All functions were designed to be easy to slot into an existing reduction pipeline (assuming the data is formatted correctly). If there are features you would like to see added to ```specsuite```, please feel free to add an issue on this repository for our developers to address!

## Installation
To install the most recent version of ```specsuite```, run the following command from your terminal...
```bash
pip install specsuite
```
OR if you would like to install a version from this repository, the run...
```bash
git clone https://github.com/Autumn10677/specsuite.git
cd specsuite
pip install .
```

## How can I test ```specsuite``` runs on my computer?

We have provided a handful of files and scripts that should help you get started on processing your data.

- ```specsuite_env.yml``` ~ A working Conda environment for the current version of the package.
- ```workflow.smk``` ~ This is a "snakemake workflow" set to run on some sample data taken from APO's long-slit spectrograph.

To run this workflow on your own computer, first clone the repository using...
```bash
git clone https://github.com/Autumn10677/specsuite.git
cd specsuite
```

Then run...
```bash
conda env create -f environment.yml
conda activate specsuite_env
snakemake --cores 1
```

This should deposit a set of files in an 'output/' folder that you can use to check out how the pipeline works at various steps in the analysis. These outputs include both images and '.npy' files used for storing exposure data between steps of the pipeline. If you see...

```bash
Finished jobid: 0 (Rule: all)
4 of 4 steps (100%) done
```

...then the pipeline ran successfully!