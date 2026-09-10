# Preprocessing extracted features

In this module, we apply preprocessing scripts to:

1. Harmonize the extracted features from SQLite to single-cell parquet profile with CytoTable.
2. Clean the single-cell profile to remove mis-segmentations and blurry cells with coSMicQC.
3. Annotate, normalize, and feature select the profile with Pycytominer.

## Perform preprocessing on data

To perform preprocessing on the profile, run the bash script [perform_preprocessing.sh](./perform_preprocessing.sh) using the command below:

```bash
source perform_preprocessing.sh
```

To run this preprocessing for the HLHS dataset instead, run each notebook (`0.convert_cytotable.ipynb`, `1.sc_quality_control.ipynb`, `2.single_cell_processing.ipynb`) manually with `hlhs_run` set to `True` (e.g. via papermill `-p hlhs_run true` or `python nbconverted/<script>.py --hlhs-run true`), since `perform_preprocessing.sh` only runs the non-HLHS dataset.
