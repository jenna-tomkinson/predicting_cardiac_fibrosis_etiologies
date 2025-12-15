#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the environment
conda activate fibrosis_eda_env

# convert Jupyter notebook(s) to script
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb

# run Python scripts for EDA (exploratory data analysis)
python nbconverted/0.check_actin_intensity.py
python nbconverted/1.UMAP.py
python nbconverted/2.cell_count_check.py
python nbconverted/4.correlation_heatmaps.py

# deactivate the conda environment
conda deactivate
# activate R environment
conda activate fibrosis_R_env

# run R script for EDA
Rscript nbconverted/feature_heatmaps_R.r

