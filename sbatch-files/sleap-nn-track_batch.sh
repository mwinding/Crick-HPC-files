#!/bin/bash
# sleap-nn version of sleap-track_batch.sh, for models trained with sleap-nn (best.ckpt). Run from the folder of videos.
# Usage: sbatch --export=MODEL="sideview",JOB="pc" sleap-nn-track_batch.sh
# optional parameters: sbatch --export=MODEL="sideview",JOB='ptcd',FRAMES="0-10",TRACKER="simple",MODELS_DIR="/path/to/models" sleap-nn-track_batch.sh
# for JOB, p = predict, t = track, c = convert .slp to .feather, and d = DBSCAN clustering (default pc: no tracking)
# TRACKER: flow (default, optical flow, as the deployed sleap-track_batch.py) or simple (iou, hungarian, window 1)
# MODELS_DIR: folder with one <name>.centroid and one <name>.centered_instance model (default: <MODEL>/active)

#SBATCH --job-name=slp-master
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=ncpu
#SBATCH --mem=300G
#SBATCH --time=72:00:00
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

ml purge
ml Anaconda3/2024.10
source /camp/apps/eb/software/Anaconda/conda.env.sh

DIR=$(pwd)

: ${FRAMES:='all'}
: ${JOB:='pc'}
: ${BATCH_SIZE:='4'}
: ${TRACKER:='flow'}
: ${MODELS_DIR:=''}
: ${SCRIPT:='/camp/lab/windingm/home/shared/Crick-HPC-files/sbatch-files/sleap-nn-track_batch.py'}

echo "model type: $MODEL"
echo "videos directory path: $DIR"
echo "jobs, p=prediction, t=track, c=convert, d=DBSCAN to feather: $JOB"
echo "frames: $FRAMES"
echo "batches: $BATCH_SIZE"
echo "tracker: $TRACKER"

conda activate /camp/lab/windingm/home/shared/conda-envs/sleap165 #use shared sleap-nn conda env on NEMO

# run python script
# save output to log file in case there is an issue
# adding -u makes sure the python_output.log is dynamically written to
cmd="python -u $SCRIPT -m $MODEL -p $DIR -j $JOB -f $FRAMES -b $BATCH_SIZE -t $TRACKER"
if [[ -n $MODELS_DIR ]]; then cmd="$cmd -d $MODELS_DIR"; fi
eval $cmd > python_output.log 2>&1
