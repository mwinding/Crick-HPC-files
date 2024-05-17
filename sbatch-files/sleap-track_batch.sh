#!/bin/bash
# Usage: ./sleap-track_batch.sh /path/to/your/folder /path/to/centroid/model /path/to/centered_instance/model

#SBATCH --job-name=slp-master
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=ncpu
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

ml purge
ml Anaconda3/2023.09-0
source /camp/apps/eb/software/Anaconda/conda.env.sh

echo "centroid model path: $CENTROID"
echo "centered instance model path: $CEN_INS"
echo "videos directory path: $DIR"
echo "skeleton parts: $PARTS"

conda activate sleap

# run python script
# save output to log file in case there is an issue
cmd="python sleap-track_batch.py -m1 "$CENTROID" -m2 "$CEN_INS" -p "$DIR" -s "$PARTS""
eval $cmd > python_output.log 2>&1