#!/bin/bash
# Usage: ./sleap-track_batch.sh /path/to/your/folder /path/to/centroid/model /path/to/centered_instance/model

# Path to the directory containing the video files
DIR=$1

# Path to the centroid model
CENTROID=$2

# Path to the centered instance model
CEN_INS=$3

# Count the number of MP4 files in the specified directory
num_files=$(find "${DIR}" -maxdepth 1 -type f -name '*.mp4' | wc -l)

# Check for the presence of mp4 files
if [ "$num_files" -eq "0" ]; then
    echo "No .mp4 files found in the directory."
    exit 1
else
    echo "Submitting $num_files jobs for processing."
fi

# Submit the job with the environment variables and job array configuration
sbatch --export=DIR="$DIR",CENTROID="$CENTROID",CEN_INS="$CEN_INS",NUM_FILES=$num_files sleap-track_array-job.sh