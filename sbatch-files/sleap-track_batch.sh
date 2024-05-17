#!/bin/bash
# Usage: ./sleap-track_batch.sh /path/to/your/folder /path/to/centroid/model /path/to/centered_instance/model

#SBATCH --job-name=sleap-master
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

echo "$CENTROID"
echo "$CEN_INS"
echo "$DIR"

conda activate sleap
cmd="python3 sleap-track_batch.py -m1 "$CENTROID" -m2 "$CEN_INS" -p "$DIR""
eval $cmd > python_output.log 2>&1