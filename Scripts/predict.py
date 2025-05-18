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
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score
)

warnings.filterwarnings("ignore", message=".*CUDA.*")
from Bio_ClinicalBERTClassifier import BioClinicalBERTClassifier

def evaluate_binary_predictions(true_labels, predictions, model_name="Model", save_path=None, plot_results=False, start_time=None, out_filename=None):
    """
    Evaluate binary classification predictions and save results in machine-readable format.
    
    Args:
        true_labels: List of ground truth labels (0 or 1)
        predictions: List of predicted labels (0 or 1)
        model_name: Name of the model for tracking
        save_path: Directory to save results and plots
        plot_results: Boolean indicating whether to plot confusion matrix
    
    Returns:
        Dictionary with metrics that can be easily converted to CSV
    """
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    # Get detailed classification report
    report = classification_report(true_labels, predictions, output_dict=True)
    
    # Create confusion matrix values for reporting
    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Print basic metrics to console
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print classification report to console
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))
    
    # Prepare results dictionary (machine-readable format)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    end_time = datetime.datetime.now()
    elapsed_time = (end_time - start_time).total_seconds() if start_time else None
    results = {
        'timestamp': timestamp,
        'time_taken': str(elapsed_time),
        'model_name': model_name,
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        # Add detailed metrics from classification report
        'precision_class0': round(report.get('0', {}).get('precision', 0), 4),
        'recall_class0': round(report.get('0', {}).get('recall', 0), 4),
        'f1_class0': round(report.get('0', {}).get('f1-score', 0), 4),
        'support_class0': int(report.get('0', {}).get('support', 0)),
        'precision_class1': round(report.get('1', {}).get('precision', 0), 4),
        'recall_class1': round(report.get('1', {}).get('recall', 0), 4),
        'f1_class1': round(report.get('1', {}).get('f1-score', 0), 4),
        'support_class1': int(report.get('1', {}).get('support', 0)),
        'macro_avg_precision': round(report.get('macro avg', {}).get('precision', 0), 4),
        'macro_avg_recall': round(report.get('macro avg', {}).get('recall', 0), 4),
        'macro_avg_f1': round(report.get('macro avg', {}).get('f1-score', 0), 4),
        'weighted_avg_precision': round(report.get('weighted avg', {}).get('precision', 0), 4),
        'weighted_avg_recall': round(report.get('weighted avg', {}).get('recall', 0), 4),
        'weighted_avg_f1': round(report.get('weighted avg', {}).get('f1-score', 0), 4)
    }

    # Save results to CSV if path is provided
    if save_path:   
        # results_file = os.path.join(save_path, f"evaluation_results.csv")
        with open(out_filename, 'w') as f:
            writer = pd.DataFrame([results])
            writer.to_csv(f, index=False, header=f.tell() == 0)
        print(f"Results saved to {out_filename}")
    
    # Save visualizations if path is provided
    if plot_results:
        # Confusion Matrix using only matplotlib
        plt.figure(figsize=(8, 6))
        
        # Create the confusion matrix plot manually
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.colorbar()
        
        # Set labels
        classes = ['Negative (0)', 'Positive (1)']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations to the cells
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Labels')
        plt.xlabel('Predicted Labels')
        
        plt.savefig(os.path.join(save_path, f"{model_name}_confusion_matrix.png"))
        plt.close()
    
    return results

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
    if args.label_column not in dataset.columns:
        print(f"Error: Label column '{args.label_column}' not found in the dataset.")
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
        true_labels=dataset[args.label_column],
        output_csv=True
    )
    predictions = predictions.tolist()  # Convert numpy array to list
    
    # Get ground truth labels
    true_labels = dataset[args.label_column].tolist()
    
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
    
    # Evaluate predictions and get machine-readable results
    results = evaluate_binary_predictions(
        true_labels=true_labels,
        predictions=predictions,
        model_name=model_name,
        save_path=save_path,
        plot_results=args.plot_results,
        start_time=start_time,
        out_filename=results_file
    )
    
    # Add additional context to results
    results['data_file'] = data_name
    results['num_samples'] = len(true_labels)
    
    # # Check if file exists to determine if we need to write headers
    # file_exists = os.path.isfile(results_file)
    
    # # Write results to CSV
    # with open(results_file, 'a' if file_exists else 'w', newline='') as f:
    #     writer = pd.DataFrame([results])
    #     writer.to_csv(f, index=False, header=not file_exists)
    
    # print(f"Results saved to {results_file}")
    
    # Return results dictionary
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to use Bio_ClinicalBERTClassifier for predictions and evaluation."
    )
    parser.add_argument("--weight_file", type=str, required=True, help="Path to model weight file (.pt)")
    parser.add_argument("--data_file", type=str, required=True, help="Path to dataset CSV file")
    parser.add_argument("--text_column", type=str, default="TEXT", help="Name of text column in the dataset")
    parser.add_argument("--label_column", type=str, default="LABEL", help="Name of label column in the dataset")
    parser.add_argument("--save_results_path", type=str, default=None, 
                        help="Path to save evaluation results and metrics")
    parser.add_argument("--primary_key", type=str, help="Primary key for the dataset")
    parser.add_argument("--plot_results", action='store_true', default=False,
                        help="Whether to plot and save confusion matrix")
    parser.add_argument("--predict", action='store_true', default=False,
                        help="Whether to run predictions on the dataset")
    parser.add_argument("--finetune", action='store_true', default=False,
                        help="Whether the model has been fine-tuned")

    args = parser.parse_args()

    main(args)