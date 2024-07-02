#!/bin/bash
# Usage: sbatch --export=MODEL="sideview",TRACK="True" sleap-track_batch.sh

# *** MAKE SURE TO USE A REMOTELY-TRAINED MODEL!!!! ***
# we have experienced issues with locally trained models running remotely...

#SBATCH --job-name=slp-master
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=ncpu
#SBATCH --mem=12G
#SBATCH --time=48:00:00
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

ml purge
ml Anaconda3/2023.09-0
source /camp/apps/eb/software/Anaconda/conda.env.sh

DIR=$(pwd)

# Set TRACK to True if it is not entered by user
: ${TRACK:='True'}

echo "model type: $MODEL"
echo "videos directory path: $DIR"
echo "track animals: $TRACK"

conda activate sleap

# run python script
# save output to log file in case there is an issue
cmd="python -u /camp/lab/windingm/home/shared/TestDev/Crick-HPC-files/sbatch-files/sleap-track_batch.py -m "$MODEL" -p "$DIR" -t "$TRACK"" # adding -u makes sure the python_output.log is dynamically written to
eval $cmd > python_output.log 2>&1