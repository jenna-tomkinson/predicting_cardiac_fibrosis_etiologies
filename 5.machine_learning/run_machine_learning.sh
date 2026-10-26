#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the environment
conda activate fibrosis_machine_learning

# set the plate to train with (defaults to the subtyping plate), e.g. PLATE_TO_TRAIN=CARD-CelIns-CX7_260407120001 source run_machine_learning.sh
export PLATE_TO_TRAIN=${PLATE_TO_TRAIN:-CARD-CelIns-CX7_260814100001}

# convert Jupyter notebook(s) to script
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb

# run Python scripts for machine learning 
python nbconverted/0.split_data.py
python nbconverted/1.train_models.py
python nbconverted/2.extract_model_performance.py
python nbconverted/4.extract_model_coef.py
python nbconverted/5.leave_one_heart_out.py
python nbconverted/6.positional_effect_platemaps.py
python nbconverted/7.multiclass_feature_importance.py
python nbconverted/8.multiclass_single_cell_montage.py
python nbconverted/9.export_multiclass_probabilities.py

# Activate R environment
conda deactivate
conda activate r_fibrosis_env

# run R script for visualizing model performance
Rscript nbconverted/3.vis_model_performance.r
