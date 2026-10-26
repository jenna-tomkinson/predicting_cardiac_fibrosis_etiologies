# Exploratory data analysis

In this module, we perform simple exploratory data analysis methods to better understand the data, including:

1. Check if the actin intensity between healthy versus failing cells looks to be accurate with the biology.
2. Generate UMAPs with the profiles that had QC performed or not to see how QC impacts the clustering.
3. Check the cell counts across hearts.
4. Generate well-level Pearson correlation heatmaps across all plates, per plate, and per heart using the bulk profiles from `3.preprocessing_profiles/3.bulk_processing.ipynb`.
5. Compare the replicate correlation per well (mean Pearson correlation with the other wells from the same heart on the plate) to the percent of single cells removed by QC per well, to find which hearts have the most consistent wells and the fewest segmentation failures.

## Perform EDA on data

To perform EDA on the profile, run the bash script [perform_eda.sh](./perform_eda.sh) using the command below:

```bash
source perform_eda.sh
```
