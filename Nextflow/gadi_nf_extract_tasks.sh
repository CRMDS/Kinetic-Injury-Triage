#!/bin/bash

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

echo "Done. Output saved to $output_file"

