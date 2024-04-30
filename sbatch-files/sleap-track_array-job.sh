#!/bin/bash
#SBATCH --job-name=rsync_pis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-"$NUM_FILES"
#SBATCH --partition=ncpu
#SBATCH --mem=10G
#SBATCH --time=08:00:00
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

ml purge
ml Anaconda3/2023.09-0
source /camp/apps/eb/software/Anaconda/conda.env.sh

conda activate sleap

# Get the directory path from the environment variable
DIR=$DIR
CENTROID=$CENTROID
CEN_INS=$CEN_INS

FILE_PATH=$(find "${DIR}" -maxdepth 1 -type f -name '*.mp4' | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")

echo "Processing file $FILE_PATH"
NAME_VAR=$(basename "$FILE_PATH" .mp4)
sleap-track "$FILE_PATH" -m $CENTROID -m $CEN_INS -o $DIR/$NAME_VAR.predictions.slp
sleap-convert $DIR/$NAME_VAR.predictions.slp -o $DIR/$NAME_VAR.json --format json
sleap-convert $DIR/$NAME_VAR.predictions.slp -o $DIR/$NAME_VAR.csv --format csv
