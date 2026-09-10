#!/bin/bash

(
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# initialize conda for this shell session and activate the environment
eval "$(conda shell.bash hook)"
conda activate fibrosis_preprocessing_env

# convert Jupyter notebook(s) to script
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb

# run Python scripts for preprocessing
python nbconverted/0.convert_cytotable.py

# run single-cell QC per plate with papermill
mkdir -p qc_notebooks
shopt -s nullglob
converted_profiles=(data/converted_profiles/*_converted.parquet)

if [ ${#converted_profiles[@]} -eq 0 ]; then
    echo "No converted profiles found in data/converted_profiles. Cannot run QC."
    exit 1
fi

for converted_profile in "${converted_profiles[@]}"; do
    plate="$(basename "$converted_profile" _converted.parquet)"
    echo "Running single-cell QC for ${plate}"
    papermill \
        1.sc_quality_control.ipynb \
        "qc_notebooks/${plate}_sc_quality_control.ipynb" \
        -p plate_name "$plate" \
        -p render_images false
done

python nbconverted/2.single_cell_processing.py
)
