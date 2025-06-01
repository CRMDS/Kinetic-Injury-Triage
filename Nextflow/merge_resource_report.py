import csv
import re
import pandas as pd

# Input files
file1_path = "gadi-nf-core-joblogs.tsv"  # Contains Log_path
file2_path = "nf-tasks.tsv"  # Base file (file_path + task_info)
param_file_path = "parameter_search.csv"  # parameters used in the pipeline
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


# Step 3: Take the merged data, remove the first column (file_path),
# for the second column, take just the task ID,
# and write the output to the same TSV file

# Read the original data
with open(output_path, newline='') as fin:
    reader = csv.reader(fin, delimiter='\t')
    rows = list(reader)

# Process and overwrite
with open(output_path, 'w', newline='') as fout:
    writer = csv.writer(fout, delimiter='\t')

    header = rows[0]
    # Drop first column, parse the second column for task name and ID
    # and write the new header 
    new_header = ['task_name', 'task_id'] + header[2:]  
    writer.writerow(new_header)

    for row in rows[1:]:
        task_info = row[1]
        match = re.match(r'(\w+)\s*\((\d+)\)', task_info)
        if match:
            task_name = match.group(1)
            task_id = match.group(2)
        else:
            task_name = 'NA'
            task_id = 'NA'
        new_row = [task_name, task_id] + row[2:]
        # match = re.search(r'\((\d+)\)', task_info)
        # task_id = match.group(1) if match else 'NA'
        # new_row = [task_id] + row[2:]
        writer.writerow(new_row)


# Step 4: Sort the output file by task_id
# Read the processed file into a DataFrame, sort by task_id, and overwrite
df = pd.read_csv(output_path, sep='\t')
df['task_id'] = pd.to_numeric(df['task_id'], errors='coerce')
df = df.sort_values('task_id').reset_index(drop=True)
df.to_csv(output_path, sep='\t', index=False)


# Step 5: Merge with the parameter file
df_params = pd.read_csv(param_file_path)
df_merged = pd.concat([df_params, df], axis=1)
# Move 'task_name' and 'task_id' to the front
df_merged = df_merged[['task_name', 'task_id'] + [col for col in df_merged.columns if col not in ['task_name', 'task_id']]]
df_merged.to_csv(output_path, sep='\t', index=False)


# Final step: Print confirmation message
print(f"Merged data written to {output_path}")
