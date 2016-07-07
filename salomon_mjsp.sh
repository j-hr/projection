#!/bin/bash
# multi job start parallel for cluster salomon
# parameter: input CSV file:
# 	on each row: name_of_job number_of_nodes number_of_cores(1-24, use only one node if this is <24) queue_name walltime_in_hours main.py your parameters
# adds automatically -n name_of_job to parameters of main.py

# QUEUES:
# see https://docs.it4i.cz/salomon/resource-allocation-and-job-execution
# qfree - max 12h  (for free, low priority, may wait befor start)
# qprod - max 48h  (charge your account)
# qlong - max 144h (charge your account)

# using # on any line omits this line from queue

grep -v "#" $1 | while read name nodes cores queue walltime run
do
    echo \#!/bin/bash > $name
    echo \#PBS -q qfree
    echo \#PBS -l select=$nodes:ncpus=$cores:mpiprocs=$cores,walltime=$walltime:00:00
    echo \#PBS -A OPEN-7-33
    echo \#PBS -o ${name}o.out
    echo \#PBS -e ${name}o.out
    echo echo $name >> $name
    echo cd /scratch/work/user/hron/WORK/projection >> $name
    echo module use /home/hron/pkg/Modules >> $name
    echo module add fenics/1.7.0dev >> $name
    echo echo Running on host \`hostname\` >> $name
    echo echo uloha: $run -n $name >> $name
    echo echo Time is \`date\` >> $name
    echo mpirun --display-map --map-by core --bind-to core python $run -n $name | tee ${name}.temp >> $name
    echo echo Time is \`date\` >> $name
    qsub $name
    sleep 2
done
