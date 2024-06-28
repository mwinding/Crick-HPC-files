#!/bin/bash
# Usage: sbatch --export=DIR="/path/to/video_folder",CENTROID="/path/to/centroid_model",CEN_INS="/path/to/centered_instance_model",PARTS="body_part1 body_part2 ..." sleap-track_batch.sh

# *** MAKE SURE TO USE A REMOTELY-TRAINED MODEL!!!! ***
# we have experienced many bugs with locally trained models running remotely...

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

echo "centroid model path: $CENTROID"
echo "centered instance model path: $CEN_INS"
echo "videos directory path: $DIR"
echo "skeleton parts: $PARTS"

conda activate sleap

# run python script
# save output to log file in case there is an issue
cmd="python -u sleap-track_batch.py -m1 "$CENTROID" -m2 "$CEN_INS" -p "$DIR" -s "$PARTS"" # adding -u makes sure the python_output.log is dynamically written to
eval $cmd > python_output.log 2>&1