#!/bin/bash
# Usage: sbatch --export=MODEL="sideview",JOB="pcd" sleap-track_batch.sh
# optional parameters: sbatch --export=MODEL="sideview",JOB='ptcd',TRACK="False",FRAMES="0-10" sleap-track_batch.sh
# for JOB, p = predict with SLEAP, t = track with SLEAP, c = convert .slp to .feather, and d = DSCAN clustering

# *** MAKE SURE TO USE A REMOTELY-TRAINED MODEL!!!! ***
# we have experienced issues with locally trained models running remotely...

#SBATCH --job-name=slp-master
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=ncpu
#SBATCH --mem=300G
#SBATCH --time=48:00:00
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

ml purge
ml Anaconda3/2023.09-0
source /camp/apps/eb/software/Anaconda/conda.env.sh

DIR=$(pwd)

# Set TRACK to True if not entered by user
: ${TRACK:='True'}

# Set FRAMES to all if not entered by user
: ${FRAMES:='all'}

# Set JOB to ptc if not entered by user; p = predict, t = track, c = convert to feather output
: ${JOB:='ptc'}

: ${BATCH_SIZE:='32'}

echo "model type: $MODEL"
echo "videos directory path: $DIR"
echo "jobs, p=prediction, t=track, c=convert to feather: $JOB"
echo "frames: $FRAMES"
echo "batches: $BATCH_SIZE"

conda activate /camp/lab/windingm/home/shared/conda-envs/sleap #use shared conda env on NEMO

# run python script
# save output to log file in case there is an issue
# adding -u makes sure the python_output.log is dynamically written to
cmd="python -u /camp/lab/windingm/home/shared/Crick-HPC-files/sbatch-files/sleap-track_batch.py -m "$MODEL" -p "$DIR" -j "$JOB" -f "$FRAMES" -b "$BATCH_SIZE"" 
eval $cmd > python_output.log 2>&1