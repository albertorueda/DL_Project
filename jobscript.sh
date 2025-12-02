#!/bin/bash
# embedded options to bsub - start with #BSUB

# -- our name ---
#BSUB -J hav_gru_partial
# -- choose queue --
#BSUB -q gpul40s

# -- specify that we need 4GB of memory per core/slot --
# so when asking for 4 cores, we are really asking for 4*4GB=16GB of memory 
# for this job. 
#BSUB -R "rusage[mem=4GB]"

# -- Notify me by email when execution begins --
#BSUB -B
# -- Notify me by email when execution ends   --
#BSUB -N

# -- email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s253057@dtu.dk

# -- Output File --
#BSUB -o Output_%J.out

# -- Error File --
#BSUB -e Output_%J.err

# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 6:00 

# -- Number of cores requested -- 
#BSUB -n 4

# -- Specify the distribution of the cores: on a single node --
#BSUB -R "span[hosts=1]"

# -- end of LSF options -- 
#BSUB -o output/big_loop_training%J.out
#BSUB -e output/big_loop_training%J.err

source dl_project/bin/activate
python3 training_big_loop.py
