#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# Load necessary modules
module load releases/2022b
module load Python/3.10.8-GCCcore-12.2.0

# Use the specific Python path
PYTHON_PATH=/opt/cecisw/arch/easybuild/2022b/software/Python/3.10.8-GCCcore-12.2.0/bin/python3

# Run your Python script with environment export
srun --export=ALL $PYTHON_PATH /home/users/a/k/akontaxa/chyperML.py ${DATASET_ID} ${ID}