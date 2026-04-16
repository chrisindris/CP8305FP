#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192   
#SBATCH --time=0-02:00:00
#SBATCH --job-name=openmp_job
#SBATCH --output=out/04_analysis_findings-%j.out

module load StdEnv/2023
module load gcc/12.3
module load python/3.11

source /scratch/indrisch/venv_cp8305fp/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export IPYTHONDIR=/scratch/indrisch/.ipython

ipython 04_analysis_findings.ipynb
#python -m ipykernel install --user --name=venv_cp8305fp --display-name="Python (venv_cp8305fp)"