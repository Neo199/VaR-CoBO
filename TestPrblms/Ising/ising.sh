#!/bin/bash -l
#SBATCH --job-name="labs"

# specify number of tasks/cores per node required
#SBATCH --array=1-10              # 20 tasks (one per instance)
#SBATCH --cpus-per-task=8         # Enough cores for Stan+GA

# specify the walltime e.g 20 mins
#SBATCH -t 520:00:00

# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niyati.seth@ucdconnect.ie

# Include the below three lines to load the intel libraries
module load intel-oneapi-compilers/2023.2.4-gcc-11.5.0-6uvfkah
module load intel-oneapi-mpi/2021.14.0-gcc-11.5.0-hjmtgxa
module load intel-oneapi-mkl/2024.2.2-gcc-11.5.0-hjitxos
module load gcc/11.5.0-gcc-11.5.0-vdl6dwy

# Setting library path to include packages rstanarm, rstan, rstantools and StanHeaders which were copied from /opt/software/R/4.4.0/lib64/R/library
#export LD_LIBRARY_PATH=/home/people/20204013/r450/lib64/R/lib:/opt/software/el9/spack/0.23/opt/spack/linux-rhel9-skylake_avx512/gcc-11.5.0/gcc-11.5.0-vdl6dwy3f2p4te5hmpfdq7muhsczmy2r/lib64:$LD_LIBRARY_PATH


# run from current directory
cd $SLURM_SUBMIT_DIR
# command to use
srun /home/people/20204013/r450/bin/Rscript ising_4x4.R $SLURM_ARRAY_TASK_ID
