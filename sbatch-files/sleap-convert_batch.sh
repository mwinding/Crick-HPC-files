#!/bin/bash
# Usage: sbatch --export=MODEL="sideview" sleap-track_batch.sh
# optional parameters: sbatch --export=MODEL="sideview",TRACK="False",FRAMES="0-10" sleap-track_batch.sh

# *** MAKE SURE TO USE A REMOTELY-TRAINED MODEL!!!! ***
# we have experienced issues with locally trained models running remotely...

#SBATCH --job-name=slp-master
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ncpu
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

ml purge
ml Anaconda3/2023.09-0
source /camp/apps/eb/software/Anaconda/conda.env.sh

DIR=$(pwd)

echo "model type: $MODEL"
echo "videos directory path: $DIR"

conda activate sleap
# conda activate /camp/lab/windingm/home/shared/conda/.../.

# run python script
# save output to log file in case there is an issue
# adding -u makes sure the python_output.log is dynamically written to
cmd="python -u /camp/lab/windingm/home/shared/TestDev/Crick-HPC-files/sbatch-files/sleap-convert_batch.py -m "$MODEL" -p "$DIR"" 
eval $cmd > python_output_slp-convert.log 2>&1