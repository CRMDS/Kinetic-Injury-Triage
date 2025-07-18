This directory contains the training, fine-tuning, and prediction results from experiments conducted on the WSU HPC, Ingham CPU along with supplementary materials and analysis scripts.

# Scripts in the directory


## Supplementary Material

`Supplementary.pdf` includes fine-tuning results on the MIMIC dataset using Adam and SGD optimizers, and predictions on the Ingham Two dataset.

`Supplementary.tex` is the LaTeX source for the supplementary PDF. 

## Model Results (CSV & XLSX)

`fine_tune_prediction_results.csv` and `fine_tune_prediction_results.xlsx` contain the model’s prediction metrics on the Ingham Two dataset after fine-tuning on the Ingham One dataset.

`fine_tune_results.csv` and `fine_tune_results.xlsx` contain the fine-tuning performance metrics on the Ingham One dataset.

`prediction_results.csv` and `prediction_results.xlsx` contain the prediction results on the Ingham One dataset using the `Bio_ClinicalBERT` model trained on the MIMIC dataset (without fine-tuning).

`training_results.csv` and `training_results.xlsx` contain the training performance results of the `Bio_ClinicalBERT` model on the MIMIC dataset.


## Timing & Performance Experiments

`time_finetune_diffCPU_AdamW_lr0.0001_dr0.15_unf1.csv` contains the fine-tuning duration results across different CPU allocations.

`time_finetune_pred_diffCPU_AdamW_lr0.0001_dr0.15_unf1.csv` contains inference time benchmarks for the same CPU sweep. 



## Statistical Analysis & Post-Processing

`resComparison.R` Paired t-tests, CI calculations, visual comparisons across conditions.

`ttestAllPairs.R` Runs all pairwise t-tests and outputs a summary table. 

`resultsAnalysis.ipynb`  Generates plots and summary stats for publication. 
