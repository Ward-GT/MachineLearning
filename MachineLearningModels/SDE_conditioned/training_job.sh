#!/bin/bash
 
#SBATCH --job-name=diffusion_train
#SBATCH --output=/home/tue/20234635/MachineLearningGit/MachineLearningModels/SDE_conditioned/results/Outputfiles/DiffusionTrain_%j.txt
#SBATCH --partition=elec.gpu-es02.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=34:00:00

# Load Modules
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load anaconda/2022.10-pth39

# Run python file
conda run -n Ward python config.py
