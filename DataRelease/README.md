We include here the MIMIC data set that we used for finetuning (Step 1 as described in the paper) the model. 

Due to user agreement, we have not included the text data here. We included the `ROW_ID`, `HADM_ID`, and our labels of the text data. The `ROW_ID` is unique to each sample in MIMIC-III NOTEEVENTS dataset, `HADM_ID` is what we used during our model training can be seen in the Slurm and Nextflow scripts. 

The Labels in the data set are: 1 for kinetic-related vehicular injury, and 0 for non-kinetic-related vehicular injury.

Note that, there are a total of 79 duplicates with 37 unique `ROW_ID` in the dataset, therefore there are 2399 unique samples in this dataset. 
