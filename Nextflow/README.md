This is the directory for Nextflow files for running and submitting the jobs onto the NCI cluster. 

# Scripts in the directory

## External code

`gadi_nfcore_report.sh` is a script from [nextflow community](https://nf-co.re/configs/nci_gadi/).
It goes through all the files under the `work` directory and create a usage report of resources used by the jobs on NCI. This script generates the `gadi-nf-core-joblogs.tsv` report that collects all the resources usages. 

Note: I had to update a few things in this script as it seems to not working with some of fields in the current NCI resource reports, also I needed to grab the GPU resource information. 


## Nextflow related

`nextflow.config` is the configuration file that contains information about gadi, `trace` is enabled in this configuration file to allow resource tracing and merging of job information later. 
See [NCI nextflow documentation](https://opus.nci.org.au/spaces/DAE/pages/138903678/Nextflow) for how to set these up for your project. 

`main.nf` is the main nextflow script that contains information about the job run. 

`pbs_nf.sh` is the submission script for NCI, use this to submit the nextflow pipeline onto the cluster to run all the other jobs. 


## Resource usage

`nf-extract-tasks.sh` is similar to the `gadi_nfcore_report.sh` script, except here we need to grab from the `.command.run` files the names of the jobs, so we can then correlate that with the actual parameters used in the jobs (from the parameter search csv file). This file just collects the path and the job name. 

`merge_resource_report.py` merges the resource report from `gadi_nfcore_report.sh` with the task id report from above. This is then merged with the trace report `nf-trace.txt` from the nextflow job to get the process name and tag. Finally, the reports are merged with the parameters file we have. This python code was tested with python3.8 and python3.9, and it only uses pandas, so should work with most python installations. 


## Others

Also included in this directory are: 

* `cleanup.sh` -- script to removed all the files generated from the nextflow pipeline, not including the files in `work` directory. 

The work directory contains all the various scripts, outputs, errors associated with each tasks in the nextflow pipeline. If the jobs run successfully, all you'll need from it are the resource usage, so just run the resource compilation code to get the final process report and then you won't need anything from it. If you want to keep them for whatever reason, then it's best to copy the whole folder into somewhere else and start with a clean directory for the next process. This way the resource gathering will be clean and only contain information from the the new process. 

**TODO** The parameters csv file is currently copied from `Slurm` directory, and a column `pid` is added to it using the following line: 
```
awk 'BEGIN {OFS=","} NR==1 {print "pid", $0; next} {print NR-1, $0}' parameter_search.csv > params_with_line.csv
```
so we can track the index of each job. Will need to rethink what we want to do here later. 


# Running the jobs via Nextflow

Before running the jobs, make sure you specify the correct processes you want to do in the `workflow` section of `main.nf`. 

To run these jobs via through Nextflow, you'll need to submit the nextflow pipeline (i.e. the `main.nf` file) as a job on the cluster. After the nextflow job start to run on the cluster, it'll spawn off the individual jobs onto the cluster, like when you submit the individual jobs with a `qsub <job>.sh` command. 

The nextflow pipeline job will only need one cpu and the minimum RAM to run, it needs to run for the entire time that all the jobs are running. So if you think your 1000 python jobs of 3 hours each will take 2 days to finish running on the cluster, then ask for 2 days for the pipeline job. Otherwise, the python jobs will not get started (or die halfway).

If all the jobs have run successfully, then you'll see mostly the following in the output of the pipeline jobs: 

```
executor >  pbspro (1)
[cb/00f2b3] process > train (2) [  0%] 0 of 10

executor >  pbspro (2)
[f3/3e2eb3] process > train (1) [  0%] 0 of 10

...

executor >  pbspro (10)
[f3/3e2eb3] process > train (1) [ 90%] 9 of 10

executor >  pbspro (10)
[96/491df7] process > train (10) [100%] 10 of 10 ✔
```

plus a few lines of resource usage information. If jobs didn't fully complete, then you'll see a lot more than these lines. 

 
