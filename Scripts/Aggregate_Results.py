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
    
    This function processes all results_summary.csv files in the Outputs directory,
    groups the results by hyperparameter configuration (excluding the random seed),
    and calculates statistics (mean, standard deviation) for each metric.
    
    Parameters
    ----------
    base_directory : str
        Path to the base directory containing the Outputs folder with test results
    
    Returns
    -------
    pandas.DataFrame or None
        DataFrame containing aggregated results with statistics, or None if no results were found
    
    Notes
    -----
    The function performs the following operations:
    1. Finds all results_summary.csv files in subdirectories of Outputs/
    2. Extracts parameters from each folder name
    3. Calculates F1 score for each individual result
    4. Groups results by hyperparameter configuration
    5. Calculates mean and standard deviation for each metric
    6. Formats the output with a consistent column order
    7. Rounds all numeric values to 5 decimal places
    """
    # Find all results_summary.csv files
    result_files = []
    for root, dirs, files in os.walk(os.path.join(base_directory, "..","Outputs"), followlinks=True):
        if "results_summary.csv" in files:
            result_files.append(os.path.join(root, "results_summary.csv"))
    
    if not result_files:
        print(f"No results found in {os.path.join(base_directory, 'Outputs')}")
        return None
    
    # Read all results into a list of dataframes
    all_results = []
    for file_path in tqdm(result_files):
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
    
    # Combine all results
    if not all_results:
        print("No valid results found")
        return None
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Define the columns to aggregate
    numeric_cols = [
        'best_val_loss', 'total_train_time_s', 'total_val_time_s',
        'accuracy', 'tn', 'fp', 'fn', 'tp', 'f1_score'
    ]
    
    # Group by the combination of parameters (excluding seed)
    group_cols = ['optimiser', 'folder_unf', 'folder_lr', 'folder_dropout']
    
    # Add a column for the number of seeds per configuration
    seed_counts = combined_df.groupby(group_cols).size().reset_index(name='num_seeds')
    
    # Calculate mean and std for each group
    result_mean = combined_df.groupby(group_cols)[numeric_cols].mean().reset_index()
    result_std = combined_df.groupby(group_cols)[numeric_cols].std().reset_index()
    
    # Rename columns in std dataframe
    for col in numeric_cols:
        result_std = result_std.rename(columns={col: f"{col}_std"})
    
    # Merge mean and std dataframes
    result = pd.merge(result_mean, result_std, on=group_cols)
    
    # Merge the seed count information
    result = pd.merge(result, seed_counts, on=group_cols)
    
    # Reorganise columns to put all means first, then all standard deviations
    # 1. Identify all mean columns and std columns
    mean_cols = numeric_cols
    std_cols = [f"{col}_std" for col in numeric_cols]
    
    # 2. Create the new column order - keeping your specified order but inserting num_seeds
    new_col_order = ["optimiser", "folder_unf", "folder_lr", "folder_dropout", "num_seeds",
                     "best_val_loss", "best_val_loss_std",
                     "total_train_time_s", "total_train_time_s_std",
                     "total_val_time_s", "total_val_time_s_std",
                     "accuracy", "accuracy_std",
                     "tn", "tn_std", "fp", "fp_std",
                     "fn", "fn_std", "tp", "tp_std",
                     "f1_score", "f1_score_std"
                    ]
    
    # 3. Reorder the columns
    result = result[new_col_order]
    
    # 4. Round all numeric columns to 5 decimal places
    numeric_columns = result.select_dtypes(include=['float64', 'float32']).columns
    result[numeric_columns] = result[numeric_columns].round(5)
    
    return result

def main():
    """
    Main function to execute the aggregation process.
    
    This function:
    1. Gets the current working directory
    2. Calls the aggregate_results function
    3. Saves the results to CSV and Excel files
    
    The output files are saved in the current directory as:
    - aggregated_results.csv
    - aggregated_results.xlsx
    """
    # Get the current directory (where the script is running)
    base_dir = os.getcwd()
    
    # Aggregate results
    result_df = aggregate_results(base_dir)
    
    if result_df is not None:
        # Save to CSV
        output_file = os.path.join(base_dir, "aggregated_results.csv")
        result_df.to_csv(output_file, index=False)
        print(f"Aggregated results saved to {output_file}")
        
        # Also save to Excel for better formatting
        excel_file = os.path.join(base_dir, "aggregated_results.xlsx")
        result_df.to_excel(excel_file, index=False)
        print(f"Aggregated results also saved to {excel_file}")

if __name__ == "__main__":
    main()