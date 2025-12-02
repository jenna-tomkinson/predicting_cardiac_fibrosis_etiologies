#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the environment
conda activate fibrosis_machine_learning

# convert Jupyter notebook(s) to script
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb

# run Python scripts for machine learning 
python nbconverted/0.split_data.py
python nbconverted/1.train_models.py
python nbconverted/2.extract_model_performance.py
python nbconverted/4.extract_model_coef.py

# Activate R environment
conda deactivate
conda activate R_fibrosis_env

# run R script for visualizing model performance
Rscript nbconverted/3.vis_model_performance.R
