# specsuite
For help with getting ```specsuite``` running on your own data, please check out our documentation (FIXME)!

## Introduction
Although other spectroscopic reduction tools exist, 

```specsuite``` was designed to help users process long-slit spectroscopic data from ___all___ ground-based telescopes, but this level of generalization is still a work in progress! As of writing, this reduction pipeline has been tested this reduction pipeline against data from...
- Gemini North (GMOS-N)
- Apache Point Observatory (KOSMOS)
- Sommers-Bausch Observatory (SBO)

In particular, ```specsuite``` aims to provide a series of modular tools that are easy to integrate into an existing workflow.

## Quickstart

To install the most recent version of ```specsuite```, run the following command from your terminal...
```bash
pip install specsuite
```
If you have an existing installation and would like to update to the most recent version, use...
```bash
pip install --upgrade specsuite
```

`specsuite` is a toolbox that can reduce data from a long-slit spectrograph.

## How can I test ```specsuite``` runs on my computer?

We have provided a handful of files and scripts that should help you get started on processing your data.

- ```specsuite_env.yml``` ~ A working Conda environment for the current version of the package.
- ```workflow.smk``` ~ This is a "snakemake workflow" set to run on some sample data taken from APO's long-slit spectrograph. 

To run this workflow on your own computer, first clone the repository using...
```bash
git clone https://github.com/Autumn10677/specsuite.git
cd specsuite
```

Then, simply run...
```bash
conda env create -f environment.yml
conda activate specsuite_env
snakemake --cores 1
```

This should deposit a set of files in an 'output/' folder that you can use to check out how the pipeline works at various steps in the analysis.