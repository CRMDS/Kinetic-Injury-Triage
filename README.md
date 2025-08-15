# Kinetic-Injury-Triage

This repository contains the full pipeline for **clinical NLP classification of kinetic injury triage notes** using Bio_ClinicalBERT.  
It includes scripts for **pretraining, fine-tuning, prediction, hyperparameter search, and statistical analysis**, designed to run across:

- **HPC (using either Slurm or PBS schedulers)**  
- **Local machine (Python scripts only)**

# Repository Structure

## `Nextflow/` 

This folder contains Nextflow pipeline files, it is currently designed for the NCI Gadi system which uses the PBS scheduler but can easily be adopted for Slurm scheduler. 

### Scripts include:

- `gadi_nfcore_report.sh` – collects resource usage from `work/` directory (adapted from [nf-core configs](https://nf-co.re/configs/nci_gadi/))  
- `gadi_nf_extract_tasks.sh` – extracts job names and `.command.run` information for correlating with parameters  
- `merge_resource_report.py` – merges resource usage with hyperparameter configurations and Nextflow trace  
- `nextflow.config` – defines PBS Pro settings for NCI Gadi queueing system  
- `main.nf` – Nextflow pipeline script managing the workflow stages  
- `pbs_nf.sh` – PBS submission script to launch `main.nf` 

### Notes:

- **NCI Gadi uses PBS Pro.**  
- Nextflow handles job orchestration, but each task is launched as a PBS job behind the scenes.  
- **GPU and CPU usage are explicitly defined in `nextflow.config` and `.nf` files.**

See `Nextflow/README.md` for more details.



## `Results/` – Metrics & Analysis

Contains **model results, timing benchmarks, and supplementary materials**:

- `fine_tune_results.csv`, `fine_tune_prediction_results.csv` – Fine-tuning metrics (Ingham One → Ingham Two)  
- `prediction_results.csv`, `training_results.csv` – Pretraining results (MIMIC dataset)  
- `Supplementary.pdf` – Summary of results (Adam/SGD comparisons)  
- `resultsAnalysis.ipynb` – Jupyter notebook for plots and statistical summaries  
- `resComparison.R`, `ttestAllPairs.R` – Paired t-tests and statistical tests  
- Timing CSVs for CPU vs GPU performance

See `Results/README.md` for more.


## `Scripts/` – Python Code (CPU/GPU)

Contains all core **Python scripts for model training, fine-tuning, and prediction**:

- `Bio_ClinicalBERTClassifier.py` – modified wrapper for `emilyalsentzer/Bio_ClinicalBERT`  
    - Supports layer freezing/unfreezing, mixed precision, per-epoch metrics, robust checkpointing  
- `train.py` – Pretraining script (MIMIC)  
- `finetune.py` – Fine-tuning script (Ingham One)  
- `predict.py` – Generates predictions and evaluation reports  
- `configCreator.py` – Generates hyperparameter search CSVs  
- `Aggregate_Results.py` – Aggregates output logs into final CSVs  
- `MIMIC3_data_preprocessing.ipynb` – Notebook for preparing MIMIC data

See `Scripts/README.md` for usage.



## `Slurm/` 

This folder contains Slurm job scripts for running the model on HPCs that use Slurm job scheduler.

### Scripts include:

- `parameter_search.slurm` – Pretraining on GPUs (MIMIC dataset). Launches 810 parallel jobs with varying hyperparameters.  
- `finetune_models.slurm` – Fine-tuning on CPUs (Ingham One dataset). Launches 240 jobs reading from `finetune_parameters.csv`.  
- `predict.slurm` – Runs predictions on fine-tuned models. Outputs per-sample predictions and metrics.

### Hyperparameter Configuration:

- `parameter_search.csv` – optimizer, learning rate, dropout, layer unfreeze, seeds (pretraining)  
- `finetune_parameters.csv` – same parameters for fine-tuning sweep  

See `Slurm/README.md` for job instructions.




# Workflow Summary

1. **Data Preparation**  
Prepare CSVs with `TEXT`, `LABEL`, and `ID` columns.

2. **Training & Fine-tuning**  
- Use `train.py` for MIMIC dataset pretraining (typically GPU)  
- Use `finetune.py` for Ingham One fine-tuning (typically CPU)

3. **Prediction & Evaluation**  
Run `predict.py` to evaluate models on Ingham Two dataset.

4. **Hyperparameter Search**  
Use `configCreator.py` + Slurm job arrays to sweep parameters.

5. **Results Aggregation**  
Run `Aggregate_Results.py` to compile final CSVs.

6. **Statistical Analysis**  
Run `resComparison.R` or `resultsAnalysis.ipynb` for plots and paired t-tests.


# Requirements

- Python 3.8–3.10  
- `transformers`, `torch`, `pandas`, `scikit-learn`, `numpy`  
- R with `ggplot2`  

See `requirements.txt` for Python packages.


# Outputs

Final outputs include:

- Model weights: `Outputs/models/bcbert_runs/` and `Outputs/models/cpu_finetune/`  
- Per-run metrics: `Outputs/models/*/results.csv`  
- Aggregated metrics: `Results/*.csv`  
- Statistical plots and comparisons


# Contact

For questions or collaboration:  
- CRMDS / Western Sydney University
- South Western Emergency Research Institute (SWERI) / Ingham Institute for Applied Medical Research
 


# License

MIT License

