import csv
import re
import pandas as pd

# Input files
file1_path = "gadi-nf-core-joblogs.tsv"  # Contains Log_path
file2_path = "gadi-nf-tasks.tsv"  # Base file (file_path + task_info)
file_trace_path = "nf-trace.txt"  # Contains task_name, task_id, tags
param_file_path = "params_with_line_test.csv"  # parameters used in the pipeline
output_path = "merged-jobreport.tsv"

# Step 1: Load file1 into a dict with stripped path as key
log_info = {}
with open(file1_path, newline='') as f1:
    reader = csv.DictReader(f1, delimiter='\t')
    for row in reader:
        base_path = row['Log_path'].replace('/.command.log', '')
        log_info[base_path] = row

# Step 2: Read file2 and merge from file1
with open(file2_path, newline='') as f2, open(output_path, 'w', newline='') as fout:
    reader = csv.DictReader(f2, delimiter='\t')
    fieldnames = reader.fieldnames + ['Service_units', 'NCPUs_used', 'CPU_time_used',
                                      'Memory_used', 'GPU_Utilisation',
                                      'GPU_Memory_Used', 'Walltime_used']
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
        writer.writerow(row)


# Step 3: Merge with the trace file, and 
# keep only the useful columns

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
df_merged.to_csv(output_path, sep='\t', index=False)


# Step 5: If there are more than one type of process, we want to create 
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
