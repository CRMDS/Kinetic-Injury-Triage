#!/bin/bash
# Script to extract 
# -- task information from .command.run files 
# -- exit code from the .exitcode file
# in a Nextflow workflow directory, and save them to csv files. 


#--------- TASK EXTRACTION ---------#

# Output CSV file
output_file="gadi-nf-tasks.tsv"

# Write CSV header
echo -e "file_path\ttask_info" > "$output_file"

# Find all .command.log files and loop through them
find work -type f -name ".command.run" | while read -r log_file; do

    # Use awk to extract line starting with # NEXTFLOW TASK: and grab everything after the colon
    awk -v file_path="$log_file" '
    /^# NEXTFLOW TASK:/ {
        split($0, parts, ": ")
        if (length(parts) > 1) {
            print file_path "\t" parts[2]
        }
    }
    ' OFS="\t"  "$log_file" >> "$output_file"

done

echo "Output saved to $output_file"


#--------- EXIT CODE EXTRACTION ---------#

# Output CSV file for exit codes
exit_code_file="gadi-nf-exitcodes.tsv"

# Write CSV header
echo -e "file_path\texit_code" > "$exit_code_file"

# Find all .exitcode files and loop through them
find work -type f -name ".exitcode" | while read -r exit_file; do

    # Extract the exit code from the file
    exit_code=$(cat "$exit_file")   

    # Write the file path and exit code to the output file
    echo -e "$exit_file\t$exit_code" >> "$exit_code_file"

done

echo "Exit codes saved to $exit_code_file"

echo "All tasks and exit codes have been extracted successfully."


