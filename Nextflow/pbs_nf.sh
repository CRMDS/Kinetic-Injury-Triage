#!/bin/bash
#PBS -l walltime=2:00:00,ncpus=1,mem=4G,wd
#PBS -l storage=scratch/mp72
#PBS -q normal
#PBS -P mp72

module load nextflow/21.04.3

nextflow run main.nf -with-trace
