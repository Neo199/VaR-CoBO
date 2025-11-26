#!/bin/bash -l
#SBATCH --job-name="Ising"
#SBATCH --array=1-10             # 20 instances
#SBATCH --cpus-per-task=22        # adjust based on how many CPUs you want each instance to use
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH -t 520:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niyati.seth@ucdconnect.ie

# Include the below three lines to load the intel libraries
module load intel-oneapi-compilers/2023.2.4-gcc-11.5.0-6uvfkah
module load intel-oneapi-mpi/2021.14.0-gcc-11.5.0-hjmtgxa
module load intel-oneapi-mkl/2024.2.2-gcc-11.5.0-hjitxos
module load gcc/11.5.0-gcc-11.5.0-vdl6dwy


# --------------------------
# Match thread settings to Slurm allocation
# --------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Also tell R to respect SLURM_CPUS_PER_TASK
export R_PARALLEL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run from current directory
cd $SLURM_SUBMIT_DIR

# Launch R with srun
srun /home/people/20204013/r450/bin/Rscript mod_ising.R $SLURM_ARRAY_TASK_ID
