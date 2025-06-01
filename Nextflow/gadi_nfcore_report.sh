#!/bin/bash

#------------------------------------------------------------------
# nfcore_gadi_usage_report/1.0
# Platform: NCI Gadi HPC
#
# Description:
# This script gathers the job requests and usage metrics from Gadi log
# files hidden in nextflow work directories for a collection of job log 
# files within the current directory, and calculates efficiency values 
# using the formula e = cputime/walltime/cpus_used.
#
# Usage:
# command line, eg:
# bash nfcore-gadi-usage-report.sh
#
# Output:
# Tab-delimited summary of the resources requested and used for each job
# will be printed to tsv file: gadi-nf-core-joblogs.tsv.
#
# Date last modified: 08/08/23
#
# If you use this script towards a publication, please acknowledge the
# Sydney Informatics Hub (or co-authorship, where appropriate).
#
# Suggested acknowledgement:
# The authors acknowledge the scientific and technical assistance
# <or e.g. bioinformatics assistance of <PERSON>> of Sydney Informatics
# Hub and resources and services from the National Computational
# Infrastructure (NCI), which is supported by the Australian Government
# with access facilitated by the University of Sydney.
#------------------------------------------------------------------
#
# Modified by Rosalind Wang, May 2025
# changes: 
# -- update CPU time used to extract the correct field. 
# -- update Memory requested and used to extract make it more robust.
# -- add extracting GPU information


# File to save the parsed results
usage_file="gadi-nf-core-joblogs.tsv"

# Initialise the result file with headers
echo -e "Log_path\tExit_status\tService_units\tNCPUs_requested\tNCPUs_used\tCPU_time_used\tMemory_requested\tMemory_used\tNGPUs_Requested\tGPU_Utilisation\tGPU_Memory_Used\tWalltime_requested\tWalltime_used\tJobFS_requested\tJobFS_used" > "$usage_file"

# Find and process .command.log files
find work -type f -name ".command.log" | while read -r log_file; do
    file_name=$(echo "$log_file")

    # Extract the information and append to usage_file
    awk '
    BEGIN {
        name= "NA"
        exit_status = "NA"
        service_units = "NA"
        ncpus_requested = "NA"
        ncpus_used = "NA"
        cpu_time_used = "NA"
        memory_requested = "NA"
        memory_used = "NA"
	ngpus_requested = "NA"
	gpu_utilisation = "NA"
	gpu_memory_used = "NA"
        walltime_requested = "NA"
        walltime_used = "NA"
        jobfs_requested = "NA"
        jobfs_used = "NA"
    }
    /=====/ {flag=!flag; next}
    flag {
        if($0 ~ /Exit Status/) exit_status = $3
        if($0 ~ /Service Units/) service_units = $3
        if($0 ~ /NCPUs Requested/) ncpus_requested = $3
        if($0 ~ /NCPUs Used/) ncpus_used = $3
        if($0 ~ /CPU Time Used/) cpu_time_used = $4
	if ($0 ~ /Memory Requested:/ && $0 ~ /Memory Used:/) {
	    match($0, /Memory Requested:[[:space:]]*([0-9.]+[A-Za-z]+)/, arr1)
	    match($0, /Memory Used:[[:space:]]*([0-9.]+[A-Za-z]+)/, arr2)
	    if (RSTART > 0) {
	        memory_requested = arr1[1]
	        memory_used = arr2[1]
	    }
	}
	# if($0 ~ /Memory Requested/) memory_requested = $3
        # if($0 ~ /Memory Used/) memory_used = $6
	if ($0 ~ /NGPUs Requested:/)  ngpus_requested = $3
	if ($0 ~ /GPU Utilisation:/)  gpu_utilisation = $6
	if ($0 ~ /GPU Memory Used:/)  gpu_memory_used = $4
        if($0 ~ /Walltime requested/) walltime_requested = $3
        if($0 ~ /Walltime Used/) walltime_used = $6
        if($0 ~ /JobFS requested/) jobfs_requested = $3
        if($0 ~ /JobFS used/) jobfs_used = $6
    }
    END {
        print "'$file_name'", exit_status, service_units, ncpus_requested, ncpus_used, cpu_time_used, memory_requested, memory_used, ngpus_requested, gpu_utilisation, gpu_memory_used, walltime_requested, walltime_used, jobfs_requested, jobfs_used
    }' OFS="\t" $log_file >> $usage_file
done

echo "Results have been parsed to $usage_file."
