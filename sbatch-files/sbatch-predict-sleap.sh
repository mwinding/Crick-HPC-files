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
source /camp/apps/eb/software/Anaconda/conda.env.sh

conda activate sleap
sleap-track videos/UNWRAPPED_2024-02-15_14-17-53_pupae_01-31_LK_AM_2_S1.mp4.jpg --video.dataset video --video.input_format channels_last -m 240306_235934.centroid -m 240306_235934.centered_instance