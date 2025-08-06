This is the directory for **Bio_ClinicalBERT scripts for training, fine-tuning, prediction, and evaluation** on clinical NLP datasets using the WSU HPC, Ingham CPU, or local machine.

# Scripts in the directory

## External code

`Bio_ClinicalBERTClassifier.py` is the core model wrapper based on `emilyalsentzer/Bio_ClinicalBERT`.  
It handles the full workflow of loading models, fine-tuning, layer freezing/unfreezing, optimizer setup, mixed-precision training, and result saving. 

Note: We had to modify a few things in this class to handle dynamic layer unfreezing, robust error handling for local model loading when HuggingFace servers are unavailable, and additional output features like per-epoch timing and evaluation CSVs.

### Early Stopping

Line 324 in `Bio_ClinicalBERTClassifier.py` should be `if val_loss < best_val - epsilon:`, where `epsilon` is a small value (e.g. 1e-4). However, to ensure reproducibility of the results in the paper, we have opted to keep the code as it is.


## Training, Fine-tuning, and Prediction

`train.py` is the training script for BioClinicalBERT on clinical text datasets (e.g., MIMIC-III). It supports:

- Fine-tuning from scratch or from pretrained weights  
- Adjustable optimizer, learning rate, dropout, and layer unfreezing  
- Saves per-epoch metrics, confusion matrices, and model checkpoints  

`finetune.py` is similar to `train.py` but designed for **secondary fine-tuning** (e.g., from MIMIC-III to Ingham data).  
It loads existing `.pt` model weights, applies new parameters, and outputs fine-tuned weights and evaluation results.

`predict.py` is the **prediction and evaluation script**.  
It loads trained or fine-tuned models and generates predictions with:

- Accuracy, precision, recall, F1-score, and confusion matrix  
- Per-sample predictions saved to CSV  
- Inference time logging for performance benchmarking  


## Hyperparameter Sweeps

`configCreator.py` creates a grid search of hyperparameter configurations for fine-tuning.  
It generates `Slurm/finetune_parameters.csv` containing combinations of:

- Optimizer types (AdamW, Adam)  
- Learning rates  
- Dropout rates  
- Number of BERT layers to unfreeze  
- Random seeds  

Note: For reproducibility, the random seeds are fixed to ensure consistent experiments across runs.


## Result Aggregation

`Aggregate_Results.py` aggregates the results from multiple experiments and outputs summary tables.

It processes:

- `results_summary.csv` (training runs)  
- `results_summary_fine_tune.csv` (fine-tuned runs)  
- `model_evaluation_results.csv` (prediction runs)  
- `model_finetuned_evaluation_results.csv` (fine-tuned prediction runs)

This script generates:

- `training_results.csv`  
- `fine_tune_results.csv`  
- `prediction_results.csv`  
- `fine_tune_prediction_results.csv`

To run: 
```
python3 Aggregate_Results.py --base_directory <path/to/KIT>
```

All outputs are saved in the `Results/` directory.

## Others

Also included in this directory are:

* `MIMIC3_data_preprocessing.ipynb` — Jupyter notebook for preprocessing MIMIC-III data into `TEXT` and `LABEL` columns for model training. This handles text cleaning, label mapping, and dataset export.

The `Outputs/` directory contains all model checkpoints, per-run CSV logs, and predictions.  

If the models train and run successfully, all you'll need from `Outputs/` is the `results_summary.csv` and `model_finetuned_evaluation_results.csv` files. If you want to keep all checkpoints and intermediate logs, it's recommended to move the folder for archiving and start a clean run for new experiments.


