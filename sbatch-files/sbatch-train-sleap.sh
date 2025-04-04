#!/bin/bash

#SBATCH --job-name=SLEAP_training
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=200G
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

conda activate /camp/lab/windingm/home/shared/conda-envs/sleap #use shared conda env on NEMO
bash train-script.sh