#!/bin/bash

#SBATCH --job-name=SLEAP_training
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --output=slurm-%j.out
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

ml purge
ml Anaconda3/2023.09-0
ml CUDA/12.2.0
source /camp/apps/eb/software/Anaconda/conda.env.sh

conda activate sleap
sleap-track /camp/lab/windingm/home/users/seggewa/ArrayVideos/2024-02-04_16-04-36_SV5.mp4 --verbosity rich --frames 1-10 -m /camp/lab/windingm/home/users/seggewa/ArrayVideos/nemo-trained_model/models/240625_220541.centroid -m /camp/lab/windingm/home/users/seggewa/ArrayVideos/nemo-trained_model/models/240625_220541.centered_instance -o /camp/lab/windingm/home/users/seggewa/ArrayVideos/nemo-trained_model/lena_test.predictions.slp