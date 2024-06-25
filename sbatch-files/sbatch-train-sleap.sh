#!/bin/bash

#SBATCH --job-name=SLEAP_training
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

ml purge
ml Anaconda3/2023.09-0
ml CUDA/12.2.0
source /camp/apps/eb/software/Anaconda/conda.env.sh

conda activate sleap
bash train-script.sh