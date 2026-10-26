# Machine learning

In this module, we train machine learning models on different heart subtypes and all hearts together.
We select one plate to train with (set with the `PLATE_TO_TRAIN` environment variable, defaulting to `CARD-CelIns-CX7_260814100001`) and generate four data splits:

1. Female heart holdout: all single cells from the female hearts (#32 and #64)
2. Well holdout: one random well per remaining heart
3. Training: 70% of the remaining single cells
4. Testing: 30% of the remaining single cells

Training and testing datasets are stratified by heart failure subtype (Healthy, DCM, and ICM).

We also train models on the HLHS x2 plate (`CARD-CelIns-CX7_260407120001`), which adds HLHS as a failing subtype (a binary Healthy vs HLHS model and a fourth class in the multi-class model).
The female hearts are not on this plate, so there is no female heart holdout and only the well holdout, training, and testing splits are made.

Hyperparameters are selected with cross validation folds grouped by heart, so that all single cells from a heart are in the same fold.
Since the testing and well holdout splits contain the same hearts used in training, we also perform a leave-one-heart-out evaluation, where each heart is held out in turn and predicted by a model trained on the remaining hearts.

To see which features the multi-class model (Healthy vs DCM vs ICM) relies on, we plot the max |coefficient| per class split by compartment, measurement type, and channel, the importance of correlation features per channel pair, and the top features for the Non-failing (Healthy) class compared with the coefficients of the same features in every class.
We also make a single-cell image montage (nuclei in cyan, F-actin in magenta, and ER in yellow) of random cells from the most confident multi-class predictions per class, using only cells not used in training, to visually check for differences between the classes.
Since the most confident ICM and DCM cells each come from a single heart, we also make a second montage with the cells picked evenly across the hearts in each class.

To check for positional effects, we generate a platemap of the mean healthy predicted probability per well from the all hearts binary model for the training plate using only the testing and holdout cells.

We export the multi-class predicted probabilities for every single cell on the training plate (`performance_metrics/<plate>/multiclass_single_cell_predicted_probabilities.parquet`) with the cell identifying metadata (plate, well, site, and nuclei object number), so they can be merged with the normalized single-cell profiles.
The `dataset` column shows the split of each cell, since the cells in the training split were used to train the model.

## Perform machine learning on data

To perform machine learning on the dataset, run the bash script [run_machine_learning.sh](./run_machine_learning.sh) using the command below:

```bash
source run_machine_learning.sh
```

To run on the HLHS x2 plate:

```bash
PLATE_TO_TRAIN=CARD-CelIns-CX7_260407120001 source run_machine_learning.sh
```
