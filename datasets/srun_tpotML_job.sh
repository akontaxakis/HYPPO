#!/bin/bash
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --mem=16G                     # Total memory for the job
#SBATCH --time=24:00:00              # Time limit (hh:mm:ss)


#Load necessary modules
module load releases/2022b
module load Python/3.10.8-GCCcore-12.2.0

# Use the specific Python path
PYTHON_PATH=/opt/sw/arch/easybuild/2022b/softwares/Python/3.10.8-GCCcore-12.2.0/bin/python3

# Run your Python script
python3 /home/users/a/k/akontaxa/tpotML.py ${DATASET_ID} ${ID} ${K} ${N}