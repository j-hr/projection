#!/bin/bash
# multi job start parallel for cluster snehurka
# parameter: input CSV file:
# 	on each row: name_of_job number_of_cores queue_name main.py your parameters
# adds automatically -n name_of_job to parameters of main.py

# using # on any line omits this line from queue

grep -v "#" $1 | while read name cores queue run
do
    echo \#!/bin/bash > $name
    echo \#SBATCH --job-name=$name >> $name
    echo \#SBATCH -N 1 >> $name
    echo \#SBATCH -n $cores >> $name
    echo \#SBATCH -p $queue >> $name
    echo \#SBATCH -o "${name}.out" >> $name
    echo echo $name >> $name
    echo module add fenics/1.6.0 >> $name
    echo echo Running on host \`hostname\` >> $name
    echo echo uloha: $run -n $name >> $name
    echo echo Time is \`date\` >> $name
    echo mpirun python $run -n $name >> $name
    echo echo Time is \`date\` >> $name
    sbatch $name
    sleep 2
done
