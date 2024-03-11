#!/bin/bash
# usage explanation:
# sbatch --export=VIDEO_PATH="/path/to/video_folder" --export=SAVE_PATH="path/to/save_folder" sbatch-unwrap.sh

#SBATCH --job-name=unwrap_videos
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

conda activate pyimagej-env
python unwrap_video_batch.py -p $VIDEO_PATH -s $SAVE_PATH