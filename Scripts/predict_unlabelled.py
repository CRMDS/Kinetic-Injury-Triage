"""
@model: Bio_ClinicalBERTClassifier.py
@script: predict
@author: Midhun Shyam (M.Shyam)
editor: Dr Kieran Luken
"""

import os
import argparse
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import datetime
import numpy as np

warnings.filterwarnings("ignore", message=".*CUDA.*")
from Bio_ClinicalBERTClassifier import BioClinicalBERTClassifier

def main(args):

    # Start a timer to measure execution time
    start_time = datetime.datetime.now()

    # Load dataset
    try:
        dataset = pd.read_csv(args.data_file)
    except Exception as e:
        print(f"Error reading data file {args.data_file}: {e}")
        return

    # Verify that the required columns exist
    if args.text_column not in dataset.columns:
        print(f"Error: Text column '{args.text_column}' not found in the dataset.")
        return

    # Get base file names for reporting
    model_name = os.path.basename(args.weight_file).replace('.pt', '')
    data_name = os.path.basename(args.data_file).replace('.csv', '')
    
    # Ensure output directory exists
    save_path = args.save_results_path or '.'
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\nPredicting with the model: {args.weight_file}")
    print(f"Using dataset: {args.data_file}")

    classifier = BioClinicalBERTClassifier(
        predict_run=args.predict,
        fine_tune_run=args.finetune,
        output_path=save_path  # Set output path when creating classifier
    )
    classifier.load_model(args.weight_file)
    
    # Get predictions
    predictions = classifier.predict(
        dataset[args.text_column],
        primary_key=dataset[args.primary_key] if args.primary_key else None,
        output_csv=True
    )
    predictions = predictions.tolist()  # Convert numpy array to list
        
    # Ensure output directory exists
    save_path = args.save_results_path
    os.makedirs(save_path, exist_ok=True)
    
    # Create or append to CSV file
    filename = "model"
    if args.finetune:
        filename += "_finetuned"
    if args.predict:
        filename += "_evaluation"   
    filename += "_results.csv"
    results_file = os.path.join(save_path, filename)

    cedric_results = pd.DataFrame({
        "ID": dataset[args.primary_key] if args.primary_key else range(len(dataset)),
        "Label":predictions
    })

    cedric_results.to_csv(results_file, index=False)

    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to use Bio_ClinicalBERTClassifier for predicting unlabelled data."
    )
    parser.add_argument("--weight_file", type=str, required=True, help="Path to model weight file (.pt)")
    parser.add_argument("--data_file", type=str, required=True, help="Path to dataset CSV file")
    parser.add_argument("--text_column", type=str, default="TEXT", help="Name of text column in the dataset")
    parser.add_argument("--save_results_path", type=str, default=None, 
                        help="Path to save evaluation results and metrics")
    parser.add_argument("--primary_key", type=str, help="Primary key for the dataset")
    parser.add_argument("--predict", action='store_true', default=False,
                        help="Whether to run predictions on the dataset")
    parser.add_argument("--finetune", action='store_true', default=False,
                        help="Whether the model has been fine-tuned")

    args = parser.parse_args()

    main(args)