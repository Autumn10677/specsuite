# specsuite
For help with getting ```specsuite``` running on your own data, please check out our documentation (FIXME)!

## Introduction
```specsuite``` was designed to help users process long-slit spectroscopic data from ___all___ ground-based telescopes. Currently, we have tested this reduction pipeline against data from...
- Gemini North (GMOS-N)
- Apache Point Observatory (KOSMOS)
- Sommers-Bausch Observatory (SBO)

Currently, this package is able to...
- Convert data from FITS images into numpy arrays
- Perform basic calibrations (e.g., bias / dark subtraction and flatfielding)

## Quickstart

To install the most recent version of ```specsuite```, run the following command from your terminal...
```
pip install specsuite
```
If you have an existing installation and would like to update to the most recent version, use...
```
pip install --upgrade specsuite
```

`specsuite` is a toolbox that can reduce data from a long-slit spectrograph.

## How can I test ```specsuite``` runs on my computer?

We have provided a handful of files and scripts that should help you get started on processing your data.

- ```specsuite_env.yml``` ~ A working Conda environment for the current version of the package.
- ```workflow.smk``` ~ This is a "snakemake workflow" set to run on some sample data taken from APO's long-slit spectrograph. 

To run this workflow on your own computer, first clone the repository using...
```
git clone (FIXME)
cd (FIXME)
```

Then, simply run...
```
conda env create -f environment.yml
conda activate specsuite_env
snakemake --cores 1
```
This should deposit a set of files in an 'output/' folder that you can use to check out how the pipeline works at various steps in the analysis.

## Can I use it on data from other telescopes?

Yes!