"""
Aggregate Results Script

This script aggregates model evaluation results across multiple trials with different random seeds.
It processes result files from model runs with various hyperparameter combinations,
calculates statistics (mean and standard deviation) for each configuration,
and outputs a consolidated summary in CSV and Excel formats.
"""

import os
import pandas as pd
from tqdm import tqdm

def extract_params_from_folder(folder_name):
    """
    Extract hyperparameters from the folder name.
    
    Parameters
    ----------
    folder_name : str
        Name of the folder containing test results. Expected format:
        "{optimiser}_lr-{learning_rate}_dropout-{dropout_rate}_unf-{unfrozen_layers}_seed-{random_seed}"
        Example: "Adam_lr-0.0001_dropout-0.15_unf-1_seed-36"
    
    Returns
    -------
    tuple
        (optimiser, params_dict) where:
        - optimiser (str): Name of the optimiser used (e.g., "Adam")
        - params_dict (dict): Dictionary of parameter name-value pairs extracted from the folder name
    """
    # Split folder name into parts
    parts = folder_name.split('_')
    
    # Extract optimiser
    optimiser = parts[0]
    
    # Extract other parameters
    params = {}
    for part in parts[1:]:
        if '-' in part:
            key, value = part.split('-', 1)
            # Convert numerical values
            try:
                value = float(value)
                # Convert to int if it's a whole number
                if value.is_integer():
                    value = int(value)
            except ValueError:
                pass
            params[key] = value
    
    return optimiser, params

def aggregate_results(base_directory):
    """
    Aggregate model evaluation results from all test folders.
    
    This function processes all results_summary.csv, results_summary_fine_tune.csv, 
    evaluation_results.csv, and model_finetuned_evaluation_results.csv files in the Outputs directory,
    and returns separate dataframes for training, fine-tuned training, prediction, and fine-tuned prediction results.
    
    Parameters
    ----------
    base_directory : str
        Path to the base directory containing the Outputs folder with test results
    
    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame) or combination of None values
        (training_results, fine_tune_results, prediction_results, fine_tune_prediction_results) 
        dataframes containing aggregated results
    """
    # Find all results files (training, fine-tune, prediction, and fine-tune prediction results)
    result_files = []
    fine_tune_files = []
    prediction_files = []
    fine_tune_prediction_files = []  # New list for fine-tuned prediction results
    
    for root, dirs, files in os.walk(os.path.join(base_directory, "Outputs"), followlinks=True):
        if "results_summary.csv" in files:
            result_files.append(os.path.join(root, "results_summary.csv"))
        if "results_summary_fine_tune.csv" in files:
            fine_tune_files.append(os.path.join(root, "results_summary_fine_tune.csv"))
        if "evaluation_results.csv" in files:
            prediction_files.append(os.path.join(root, "evaluation_results.csv"))
        if "model_finetuned_evaluation_results.csv" in files:  # New condition for fine-tuned prediction results
            fine_tune_prediction_files.append(os.path.join(root, "model_finetuned_evaluation_results.csv"))
    
    if not result_files and not fine_tune_files:
        print(f"No training or fine-tuning results found in {os.path.join(base_directory, 'Outputs')}")
        return None, None, None, None
    
    # Process training results
    all_results = []
    for file_path in tqdm(result_files, desc="Processing training results"):
        folder_name = os.path.basename(os.path.dirname(file_path))
        optimiser, params = extract_params_from_folder(folder_name)
        
        try:
            df = pd.read_csv(file_path)
            # Add parameters from folder name
            df['folder_optimiser'] = optimiser
            for key, value in params.items():
                df[f'folder_{key}'] = value
            
            # Calculate F1 score for each individual result
            df['f1_score'] = 2 * df['tp'] / (2 * df['tp'] + df['fp'] + df['fn'])
            
            all_results.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Process fine-tune results
    all_fine_tune_results = []
    for file_path in tqdm(fine_tune_files, desc="Processing fine-tune results"):
        folder_name = os.path.basename(os.path.dirname(file_path))
        optimiser, params = extract_params_from_folder(folder_name)
        
        try:
            df = pd.read_csv(file_path)
            # Add parameters from folder name
            df['folder_optimiser'] = optimiser
            for key, value in params.items():
                df[f'folder_{key}'] = value
            
            # Calculate F1 score for each individual result if not already present
            if 'f1_score' not in df.columns and all(col in df.columns for col in ['tp', 'fp', 'fn']):
                df['f1_score'] = 2 * df['tp'] / (2 * df['tp'] + df['fp'] + df['fn'])
            
            all_fine_tune_results.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Process prediction results
    all_predictions = []
    for file_path in tqdm(prediction_files, desc="Processing prediction results"):
        folder_name = os.path.basename(os.path.dirname(file_path))
        optimiser, params = extract_params_from_folder(folder_name)
        
        try:
            df = pd.read_csv(file_path)
            # Add parameters from folder name
            df['folder_optimiser'] = optimiser
            for key, value in params.items():
                df[f'folder_{key}'] = value
            
            # Convert prediction_time to float if it's a string
            if 'prediction_time' in df.columns and df['prediction_time'].dtype == 'object':
                df['prediction_time'] = df['prediction_time'].astype(float)
            
            all_predictions.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Process fine-tuned prediction results (new)
    all_fine_tune_predictions = []
    for file_path in tqdm(fine_tune_prediction_files, desc="Processing fine-tuned prediction results"):
        folder_name = os.path.basename(os.path.dirname(file_path))
        optimiser, params = extract_params_from_folder(folder_name)
        
        try:
            df = pd.read_csv(file_path)
            # Add parameters from folder name
            df['folder_optimiser'] = optimiser
            for key, value in params.items():
                df[f'folder_{key}'] = value
            
            # Convert prediction_time to float if it's a string
            if 'prediction_time' in df.columns and df['prediction_time'].dtype == 'object':
                df['prediction_time'] = df['prediction_time'].astype(float)
            
            # Normalize column names if needed
            if 'f1_score' in df.columns and 'f1' not in df.columns:
                df = df.rename(columns={'f1_score': 'f1'})
            elif 'f1' in df.columns and 'f1_score' not in df.columns:
                df = df.rename(columns={'f1': 'f1_score'})
            
            all_fine_tune_predictions.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Combine all results
    combined_training_df = None if not all_results else pd.concat(all_results, ignore_index=True)
    combined_fine_tune_df = None if not all_fine_tune_results else pd.concat(all_fine_tune_results, ignore_index=True)
    combined_predictions_df = None if not all_predictions else pd.concat(all_predictions, ignore_index=True)
    combined_fine_tune_predictions_df = None if not all_fine_tune_predictions else pd.concat(all_fine_tune_predictions, ignore_index=True)
    
    # Print status messages
    if not all_results: print("No valid training results found")
    if not all_fine_tune_results: print("No valid fine-tune results found")
    if not all_predictions: print("No valid prediction results found")
    if not all_fine_tune_predictions: print("No valid fine-tuned prediction results found")
    
    # Define the columns to aggregate for training
    training_numeric_cols = [
        'best_val_loss', 'total_train_time_s', 'total_val_time_s',
        'accuracy', 'tn', 'fp', 'fn', 'tp', 'f1_score'
    ]
    
    # Group results by configuration (excluding seed)
    group_cols = ['folder_optimiser', 'folder_unf', 'folder_lr', 'folder_dropout']
    
    # Process initial training results
    training_result = None
    if combined_training_df is not None:
        # Add a column for the number of seeds per configuration
        train_seed_counts = combined_training_df.groupby(group_cols).size().reset_index(name='num_seeds')
        
        # Calculate training mean and std for each group
        available_training_cols = [col for col in training_numeric_cols if col in combined_training_df.columns]
        
        training_mean = combined_training_df.groupby(group_cols)[available_training_cols].mean().reset_index()
        training_std = combined_training_df.groupby(group_cols)[available_training_cols].std().reset_index()
        
        # Rename columns in std dataframe for training results
        for col in available_training_cols:
            training_std = training_std.rename(columns={col: f"{col}_std"})
        
        # Merge training mean and std dataframes
        training_result = pd.merge(training_mean, training_std, on=group_cols)
        
        # Merge the seed count information
        training_result = pd.merge(training_result, train_seed_counts, on=group_cols)
        
        # Rename the optimiser column for clarity
        training_result = training_result.rename(columns={'folder_optimiser': 'optimiser'})
    
    # Process fine-tune results
    fine_tune_result = None
    if combined_fine_tune_df is not None:
        # Add a column for the number of seeds per configuration
        fine_tune_seed_counts = combined_fine_tune_df.groupby(group_cols).size().reset_index(name='num_seeds')
        
        # Calculate fine-tune mean and std for each group
        available_fine_tune_cols = [col for col in training_numeric_cols if col in combined_fine_tune_df.columns]
        
        fine_tune_mean = combined_fine_tune_df.groupby(group_cols)[available_fine_tune_cols].mean().reset_index()
        fine_tune_std = combined_fine_tune_df.groupby(group_cols)[available_fine_tune_cols].std().reset_index()
        
        # Rename columns in std dataframe for fine-tune results
        for col in available_fine_tune_cols:
            fine_tune_std = fine_tune_std.rename(columns={col: f"{col}_std"})
        
        # Merge fine-tune mean and std dataframes
        fine_tune_result = pd.merge(fine_tune_mean, fine_tune_std, on=group_cols)
        
        # Merge the seed count information
        fine_tune_result = pd.merge(fine_tune_result, fine_tune_seed_counts, on=group_cols)
        
        # Rename the optimiser column for clarity
        fine_tune_result = fine_tune_result.rename(columns={'folder_optimiser': 'optimiser'})
    
    # Define prediction metrics to aggregate
    prediction_metrics = [
        'prediction_time', 'accuracy', 'precision', 'recall', 'f1_score',
        'true_positive', 'false_positive', 'true_negative', 'false_negative',
        'precision_class0', 'recall_class0', 'f1_class0', 'support_class0',
        'precision_class1', 'recall_class1', 'f1_class1', 'support_class1',
        'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1',
        'weighted_avg_precision', 'weighted_avg_recall', 'weighted_avg_f1'
    ]
    
    # Process prediction results
    prediction_result = None
    if combined_predictions_df is not None and len(combined_predictions_df) > 0:
        print(f"Found {len(all_predictions)} prediction result files")
        
        # Filter to only metrics that actually exist in the data
        prediction_numeric_cols = [col for col in prediction_metrics if col in combined_predictions_df.columns]
        
        # Count prediction seeds per configuration
        pred_seed_counts = combined_predictions_df.groupby(group_cols).size().reset_index(name='num_seeds')
        
        # Calculate prediction mean and std for each group
        prediction_mean = combined_predictions_df.groupby(group_cols)[prediction_numeric_cols].mean().reset_index()
        prediction_std = combined_predictions_df.groupby(group_cols)[prediction_numeric_cols].std().reset_index()
        
        # Rename columns in std dataframe for prediction results
        for col in prediction_numeric_cols:
            prediction_std = prediction_std.rename(columns={col: f"{col}_std"})
        
        # Merge prediction mean and std dataframes
        prediction_result = pd.merge(prediction_mean, prediction_std, on=group_cols)
        
        # Merge the seed count information
        prediction_result = pd.merge(prediction_result, pred_seed_counts, on=group_cols)
        
        # Rename the optimiser column for clarity
        prediction_result = prediction_result.rename(columns={'folder_optimiser': 'optimiser'})
    
    # Process fine-tuned prediction results
    fine_tune_prediction_result = None
    if combined_fine_tune_predictions_df is not None and len(combined_fine_tune_predictions_df) > 0:
        print(f"Found {len(all_fine_tune_predictions)} fine-tuned prediction result files")
        
        # Filter to only metrics that actually exist in the data
        fine_tune_prediction_numeric_cols = [col for col in prediction_metrics 
                                           if col in combined_fine_tune_predictions_df.columns]
        
        # Count fine-tuned prediction seeds per configuration
        fine_tune_pred_seed_counts = combined_fine_tune_predictions_df.groupby(group_cols).size().reset_index(name='num_seeds')
        
        # Calculate fine-tuned prediction mean and std for each group
        fine_tune_prediction_mean = combined_fine_tune_predictions_df.groupby(group_cols)[fine_tune_prediction_numeric_cols].mean().reset_index()
        fine_tune_prediction_std = combined_fine_tune_predictions_df.groupby(group_cols)[fine_tune_prediction_numeric_cols].std().reset_index()
        
        # Rename columns in std dataframe for fine-tuned prediction results
        for col in fine_tune_prediction_numeric_cols:
            fine_tune_prediction_std = fine_tune_prediction_std.rename(columns={col: f"{col}_std"})
        
        # Merge fine-tuned prediction mean and std dataframes
        fine_tune_prediction_result = pd.merge(fine_tune_prediction_mean, fine_tune_prediction_std, on=group_cols)
        
        # Merge the seed count information
        fine_tune_prediction_result = pd.merge(fine_tune_prediction_result, fine_tune_pred_seed_counts, on=group_cols)
        
        # Rename the optimiser column for clarity
        fine_tune_prediction_result = fine_tune_prediction_result.rename(columns={'folder_optimiser': 'optimiser'})

    # Format the training results
    if training_result is not None:
        # Create the column order for training results
        training_col_order = ["optimiser", "folder_unf", "folder_lr", "folder_dropout", "num_seeds"]
        
        # Define explicit ordering for each training metric and its standard deviation
        training_metric_order = [
            "best_val_loss", "best_val_loss_std",
            "total_train_time_s", "total_train_time_s_std",
            "total_val_time_s", "total_val_time_s_std",
            "accuracy", "accuracy_std",
            "tn", "tn_std", 
            "fp", "fp_std",
            "fn", "fn_std", 
            "tp", "tp_std",
            "f1_score", "f1_score_std"
        ]
        
        # Add metrics to training_col_order only if they exist in the result dataframe
        for metric in training_metric_order:
            if metric in training_result.columns:
                training_col_order.append(metric)
        
        # Include any remaining training columns not explicitly specified
        remaining_train_cols = [col for col in training_result.columns if col not in training_col_order]
        if remaining_train_cols:
            training_col_order.extend(remaining_train_cols)
        
        # Reorder the training columns
        training_result = training_result[training_col_order]
        
        # Round all numeric columns to 5 decimal places
        training_numeric_cols = training_result.select_dtypes(include=['float64', 'float32']).columns
        training_result[training_numeric_cols] = training_result[training_numeric_cols].round(5)
    
    # Format the fine-tune results
    if fine_tune_result is not None:
        # Create the column order for fine-tune results (same as training)
        fine_tune_col_order = ["optimiser", "folder_unf", "folder_lr", "folder_dropout", "num_seeds"]
        
        # Define explicit ordering for each fine-tune metric and its standard deviation (same as training)
        fine_tune_metric_order = [
            "best_val_loss", "best_val_loss_std",
            "total_train_time_s", "total_train_time_s_std",
            "total_val_time_s", "total_val_time_s_std",
            "accuracy", "accuracy_std",
            "tn", "tn_std", 
            "fp", "fp_std",
            "fn", "fn_std", 
            "tp", "tp_std",
            "f1_score", "f1_score_std"
        ]
        
        # Add metrics to fine_tune_col_order only if they exist in the result dataframe
        for metric in fine_tune_metric_order:
            if metric in fine_tune_result.columns:
                fine_tune_col_order.append(metric)
        
        # Include any remaining fine-tune columns not explicitly specified
        remaining_fine_tune_cols = [col for col in fine_tune_result.columns if col not in fine_tune_col_order]
        if remaining_fine_tune_cols:
            fine_tune_col_order.extend(remaining_fine_tune_cols)
        
        # Reorder the fine-tune columns
        fine_tune_result = fine_tune_result[fine_tune_col_order]
        
        # Round all numeric columns to 5 decimal places
        fine_tune_numeric_cols = fine_tune_result.select_dtypes(include=['float64', 'float32']).columns
        fine_tune_result[fine_tune_numeric_cols] = fine_tune_result[fine_tune_numeric_cols].round(5)
    
    # Prediction column order and metrics
    prediction_col_order = ["optimiser", "folder_unf", "folder_lr", "folder_dropout", "num_seeds"]
    prediction_metric_order = [
        "prediction_time", "prediction_time_std",
        "accuracy", "accuracy_std",
        "precision", "precision_std",
        "recall", "recall_std",
        "f1_score", "f1_score_std",
        "true_positive", "true_positive_std", 
        "false_positive", "false_positive_std",
        "true_negative", "true_negative_std", 
        "false_negative", "false_negative_std",
        
        # Class 0 metrics
        "precision_class0", "precision_class0_std",
        "recall_class0", "recall_class0_std",
        "f1_class0", "f1_class0_std",
        "support_class0", "support_class0_std",
        
        # Class 1 metrics
        "precision_class1", "precision_class1_std",
        "recall_class1", "recall_class1_std",
        "f1_class1", "f1_class1_std",
        "support_class1", "support_class1_std",
        
        # Average metrics
        "macro_avg_precision", "macro_avg_precision_std",
        "macro_avg_recall", "macro_avg_recall_std",
        "macro_avg_f1", "macro_avg_f1_std",
        "weighted_avg_precision", "weighted_avg_precision_std",
        "weighted_avg_recall", "weighted_avg_recall_std",
        "weighted_avg_f1", "weighted_avg_f1_std"
    ]
    
    # Format the prediction results
    if prediction_result is not None:
        # Add metrics to prediction_col_order only if they exist
        for metric in prediction_metric_order:
            if metric in prediction_result.columns:
                prediction_col_order.append(metric)
        
        # Include any remaining prediction columns
        remaining_pred_cols = [col for col in prediction_result.columns if col not in prediction_col_order]
        if remaining_pred_cols:
            prediction_col_order.extend(remaining_pred_cols)
        
        # Reorder the prediction columns
        prediction_result = prediction_result[prediction_col_order]
        
        # Round all numeric columns to 5 decimal places
        prediction_numeric_cols = prediction_result.select_dtypes(include=['float64', 'float32']).columns
        prediction_result[prediction_numeric_cols] = prediction_result[prediction_numeric_cols].round(5)

        # Check which columns exist before renaming
        rename_mapping = {}
        if 'true_positive' in prediction_result.columns: rename_mapping['true_positive'] = 'tp'
        if 'false_positive' in prediction_result.columns: rename_mapping['false_positive'] = 'fp'
        if 'true_negative' in prediction_result.columns: rename_mapping['true_negative'] = 'tn'
        if 'false_negative' in prediction_result.columns: rename_mapping['false_negative'] = 'fn'
        if 'true_positive_std' in prediction_result.columns: rename_mapping['true_positive_std'] = 'tp_std'
        if 'false_positive_std' in prediction_result.columns: rename_mapping['false_positive_std'] = 'fp_std'
        if 'true_negative_std' in prediction_result.columns: rename_mapping['true_negative_std'] = 'tn_std'
        if 'false_negative_std' in prediction_result.columns: rename_mapping['false_negative_std'] = 'fn_std'

        # Only rename if columns exist
        if rename_mapping:
            prediction_result = prediction_result.rename(columns=rename_mapping)
    
    # Format the fine-tuned prediction results (new)
    if fine_tune_prediction_result is not None:
        # Add metrics to fine_tune_prediction_col_order only if they exist
        # fine_tune_prediction_col_order = prediction_col_order.copy()  # Start with the same order as regular predictions

        fine_tune_prediction_col_order = ["optimiser", "folder_unf", "folder_lr", "folder_dropout", "num_seeds"]
        fine_tune_prediction_metric_order = [
            "time_taken", "time_taken_std",
            "accuracy", "accuracy_std",
            "precision", "precision_std",
            "recall", "recall_std",
            "f1", "f1_std",  # Changed from f1_score to f1 for fine-tuned models
            "true_positive", "true_positive_std", 
            "false_positive", "false_positive_std",
            "true_negative", "true_negative_std", 
            "false_negative", "false_negative_std",
            
            # Class 0 metrics
            "precision_class0", "precision_class0_std",
            "recall_class0", "recall_class0_std",
            "f1_class0", "f1_class0_std",
            "support_class0", "support_class0_std",
            
            # Class 1 metrics
            "precision_class1", "precision_class1_std",
            "recall_class1", "recall_class1_std",
            "f1_class1", "f1_class1_std",
            "support_class1", "support_class1_std",
            
            # Average metrics
            "macro_avg_precision", "macro_avg_precision_std",
            "macro_avg_recall", "macro_avg_recall_std",
            "macro_avg_f1", "macro_avg_f1_std",
            "weighted_avg_precision", "weighted_avg_precision_std",
            "weighted_avg_recall", "weighted_avg_recall_std",
            "weighted_avg_f1", "weighted_avg_f1_std"
        ]

        
        for metric in fine_tune_prediction_metric_order:
            if metric in fine_tune_prediction_result.columns and metric not in fine_tune_prediction_col_order:
                fine_tune_prediction_col_order.append(metric)
        
        # Include any remaining fine-tuned prediction columns
        remaining_fine_tune_pred_cols = [col for col in fine_tune_prediction_result.columns 
                                          if col not in fine_tune_prediction_col_order]
        if remaining_fine_tune_pred_cols:
            fine_tune_prediction_col_order.extend(remaining_fine_tune_pred_cols)
        
        # Reorder the fine-tuned prediction columns
        # fine_tune_prediction_result = fine_tune_prediction_result[fine_tune_prediction_col_order]
        
        # Round all numeric columns to 5 decimal places
        fine_tune_prediction_numeric_cols = fine_tune_prediction_result.select_dtypes(
            include=['float64', 'float32']).columns
        fine_tune_prediction_result[fine_tune_prediction_numeric_cols] = fine_tune_prediction_result[
            fine_tune_prediction_numeric_cols].round(5)

        # Check which columns exist before renaming (same as for regular predictions)
        # rename_mapping = {}
        # if 'true_positive' in fine_tune_prediction_result.columns: rename_mapping['true_positive'] = 'tp'
        # if 'false_positive' in fine_tune_prediction_result.columns: rename_mapping['false_positive'] = 'fp'
        # if 'true_negative' in fine_tune_prediction_result.columns: rename_mapping['true_negative'] = 'tn'
        # if 'false_negative' in fine_tune_prediction_result.columns: rename_mapping['false_negative'] = 'fn'
        # if 'true_positive_std' in fine_tune_prediction_result.columns: rename_mapping['true_positive_std'] = 'tp_std'
        # if 'false_positive_std' in fine_tune_prediction_result.columns: rename_mapping['false_positive_std'] = 'fp_std'
        # if 'true_negative_std' in fine_tune_prediction_result.columns: rename_mapping['true_negative_std'] = 'tn_std'
        # if 'false_negative_std' in fine_tune_prediction_result.columns: rename_mapping['false_negative_std'] = 'fn_std'

        # # Only rename if columns exist
        # if rename_mapping:
        #     fine_tune_prediction_result = fine_tune_prediction_result.rename(columns=rename_mapping)

    return training_result, fine_tune_result, prediction_result, fine_tune_prediction_result

def main():
    """
    Main function to execute the aggregation process.
    
    This function:
    1. Gets the current working directory
    2. Calls the aggregate_results function
    3. Saves training, fine-tuned, prediction, and fine-tuned prediction results to separate CSV and Excel files
    """
    # Get the current directory (where the script is running)
    base_dir = os.getcwd()
    
    # Create Results directory if it doesn't exist
    results_dir = os.path.join(base_dir, "Results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Aggregate results - now includes fine-tuning prediction results
    training_df, fine_tune_df, prediction_df, fine_tune_prediction_df = aggregate_results(base_dir)
    
    # Save training results if available
    if training_df is not None:
        train_csv = os.path.join(results_dir, "training_results.csv")
        train_excel = os.path.join(results_dir, "training_results.xlsx")
        training_df.to_csv(train_csv, index=False)
        training_df.to_excel(train_excel, index=False)
        print(f"Training metrics saved to {train_csv} and {train_excel}")
    else:
        print("No training results to save")
    
    # Save fine-tuned training results if available
    if fine_tune_df is not None:
        fine_tune_csv = os.path.join(results_dir, "fine_tune_results.csv")
        fine_tune_excel = os.path.join(results_dir, "fine_tune_results.xlsx")
        fine_tune_df.to_csv(fine_tune_csv, index=False)
        fine_tune_df.to_excel(fine_tune_excel, index=False)
        print(f"Fine-tune metrics saved to {fine_tune_csv} and {fine_tune_excel}")
    else:
        print("No fine-tune results to save")
    
    # Save prediction results if available
    if prediction_df is not None:
        pred_csv = os.path.join(results_dir, "prediction_results.csv")
        pred_excel = os.path.join(results_dir, "prediction_results.xlsx")
        prediction_df.to_csv(pred_csv, index=False)
        prediction_df.to_excel(pred_excel, index=False)
        print(f"Prediction metrics saved to {pred_csv} and {pred_excel}")
    else:
        print("No prediction results to save")
    
    # Save fine-tuned prediction results if available (new)
    if fine_tune_prediction_df is not None:
        fine_tune_pred_csv = os.path.join(results_dir, "fine_tune_prediction_results.csv")
        fine_tune_pred_excel = os.path.join(results_dir, "fine_tune_prediction_results.xlsx")
        fine_tune_prediction_df.to_csv(fine_tune_pred_csv, index=False)
        fine_tune_prediction_df.to_excel(fine_tune_pred_excel, index=False)
        print(f"Fine-tuned prediction metrics saved to {fine_tune_pred_csv} and {fine_tune_pred_excel}")
    else:
        print("No fine-tuned prediction results to save")

if __name__ == "__main__":
    main()