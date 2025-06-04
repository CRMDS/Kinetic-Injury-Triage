#!/bin/bash

# delete all of our submitted/running jobs, or those with sched ids above $1

schedIds=$(qstat -u $USER | sed -n '6,$p' | sed 's/\..*//')

for schedId in $schedIds; do
    echo "qdel-ing $schedId"
    qdel $schedId
done
    
