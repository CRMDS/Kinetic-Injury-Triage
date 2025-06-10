import csv
import re
import pandas as pd


# Input files
resourceLog_path = "gadi-nf-core-joblogs.tsv"  # Contains Log_path
taskinfo_path = "gadi-nf-tasks.tsv"  # Base file (file_path + task_info)
file_trace_path = "nf-trace.txt"  # Contains task_name, task_id, tags
exitcode_file_path = "gadi-nf-exitcodes.tsv"
param_file_path = "params_with_line.csv"  # parameters used in the pipeline
output_path = "merged-jobreport.tsv"


# Step 1.1: Load file with resource logs into a dict with stripped path as key
log_info = {}
with open(resourceLog_path, newline='') as f1:
    reader = csv.DictReader(f1, delimiter='\t')
    for row in reader:
        base_path = row['Log_path'].replace('/.command.log', '')
        log_info[base_path] = row

# Step 1.2: Read the exitcode file 
exitcode_info = {}
with open(exitcode_file_path, newline='') as exitcode_file:
    exitcode_reader = csv.DictReader(exitcode_file, delimiter='\t')
    for row in exitcode_reader:
        # Extract the base path from the file_path
        base_path = row['file_path'].replace('/.exitcode', '')
        exitcode_info[base_path] = row['exit_code']



# Step 2: Read data about task names and merge with resource logs and exit codes
with open(taskinfo_path, newline='') as f2, open(output_path, 'w', newline='') as fout:
    reader = csv.DictReader(f2, delimiter='\t')
    fieldnames = reader.fieldnames + ['Service_units', 'NCPUs_used', 'CPU_time_used',
                                      'Memory_used', 'GPU_Utilisation',
                                      'GPU_Memory_Used', 'Walltime_used', 
                                      'Exit_status', 'exit_code']
    writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()

    for row in reader:
        base_path = row['file_path'].replace('/.command.run', '')
        log_row = log_info.get(base_path, {})
        row['Service_units'] = log_row.get('Service_units', 'NA')
        row['NCPUs_used'] = log_row.get('NCPUs_used', 'NA')
        row['CPU_time_used'] = log_row.get('CPU_time_used', 'NA')
        row['Memory_used'] = log_row.get('Memory_used', 'NA')
        row['GPU_Utilisation'] = log_row.get('GPU_Utilisation', 'NA')
        row['GPU_Memory_Used'] = log_row.get('GPU_Memory_Used', 'NA')
        row['Walltime_used'] = log_row.get('Walltime_used', 'NA')
        row['Exit_status'] = log_row.get('Exit_status', 'NA')
        exit_row = exitcode_info.get(base_path, {})
        row['exit_code'] = exit_row if isinstance(exit_row, str) else 'NA'
        writer.writerow(row)




# Step 3: Merge into the trace file, and 
# keep only the useful columns from the trace file
# Also remove the filepath and task name columns as they are not needed anymore

# Read the trace file
pd_trace = pd.read_csv(file_trace_path, sep='\t')
# Keep only the columns of "process", "name", "tag"
pd_trace = pd_trace[['process', 'name', 'tag']]

# Merge with the output file
df = pd.read_csv(output_path, sep='\t')
# remove the first column (file_path)
df = df.drop(columns=['file_path'])
# Right join the trace data, which has "name", where the output file has "task_info"
df = pd.merge(pd_trace, df, left_on ='name', right_on='task_info')


# Remove the columns "name" and "task_info" from the merged DataFrame
df = df.drop(columns=['name', 'task_info'])



# Step 4: Merge with the parameter file

# convert the tag column to numeric. 
df['tag'] = pd.to_numeric(df['tag'], errors='coerce')

# Read the parameter file
df_params = pd.read_csv(param_file_path)
df_merged = pd.merge(df_params, df, left_on='pid', right_on='tag')

# Move 'process' and 'tag' to the front
df_merged = df_merged[['process', 'tag'] + [col for col in df_merged.columns if col not in ['process', 'tag']]]
# Remove the 'pid' column
df_merged = df_merged.drop(columns=['pid'])

# Ensure the exit status columns are integers
df_merged['Exit_status'] = pd.to_numeric(df_merged['Exit_status'], errors='coerce').fillna(0).astype(int)
df_merged['exit_code'] = pd.to_numeric(df_merged['exit_code'], errors='coerce').fillna(0).astype(int)


# Step 4.2: if we have more than just 0 in the exit_code column, we want to
# save the data to a separate file with `raw` in the file name
if df_merged['exit_code'].nunique() > 1:
    raw_output_path = output_path.replace('.tsv', '_raw.tsv')
    df_merged.to_csv(raw_output_path, sep='\t', index=False)
    print(f"Raw data with multiple exit codes written to {raw_output_path}")
    # and calculate the total service units used and print to console
    total_service_units = df_merged['Service_units'].sum()
    print(f"Total service units used (raw): {total_service_units}")



# Step 5: If we have more than one exit code, we want to remove all rows with non-zero exit codes. 
# 
# TODO: When we continue the training process, instead of restarting, we need to add all the 
# rows for a single task into one row. 

# Check if there are any non-zero exit codes
if df_merged['exit_code'].nunique() > 1:
    # Filter out rows with non-zero exit codes
    df_merged = df_merged[df_merged['exit_code'] == 0]
    print("Filtered out rows with non-zero exit codes.")

# Write the merged DataFrame to a TSV file
df_merged.to_csv(output_path, sep='\t', index=False)
# Print the total service units used
total_service_units = df_merged['Service_units'].sum()
print(f"Total service units used (completed jobs): {total_service_units}")


# for debugging purpose, when I was testing the script step by step. 
# import sys
# sys.exit(0)



# Step 6: If there are more than one type of process, we want to create 
# a separate file for each type of process 
process_types = df_merged['process'].unique()
for process in process_types:
    # Filter the DataFrame for the current process type
    df_process = df_merged[df_merged['process'] == process]

    # Remove the columns "process" and "tag" from the filtered DataFrame
    df_process = df_process.drop(columns=['process', 'tag'])
    
    # Define the output file name based on the process type
    output_file = f"{process}_merged_jobreport.tsv"
    
    # Write the filtered DataFrame to a new TSV file
    df_process.to_csv(output_file, sep='\t', index=False)
    
    print(f"Data for process '{process}' written to {output_file}")


# Final step: Print confirmation message
print(f"Full merged data written to {output_path}")
