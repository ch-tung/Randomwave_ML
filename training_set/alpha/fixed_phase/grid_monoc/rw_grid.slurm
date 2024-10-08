#!/bin/bash
#SBATCH -A MAT226
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -J rw_grid

module purge
module load gcc/10.3.0 openmpi/4.1.2
module load miniforge3
source activate torch

# Loop over values of P from 0 to 9
for P in {0..9}
do
    # Run the Python script with the current value of P
    srun -N1 -n1 --exclusive python ./test_rw_SQ_mo_P.py -P ${P} &
done

# Wait for all background jobs to finish
wait