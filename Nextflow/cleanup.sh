#!/bin/bash
#
# Script to clean up all the various files generated after running and submitting nextflow jobs
# Also the logs in the PBS_Logs directory (from stdout and stderr of python) are deleted. 

# TODO: add a user input here to ensure all logs that we want to move have been moved. 

rm .nextflow.log*
rm *.sh.e*
rm *.sh.o*
rm ../PBS_Logs/*
