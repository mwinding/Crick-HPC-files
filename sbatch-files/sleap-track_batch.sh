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

# Create SBATCH script dynamically
echo "#!/bin/bash
#SBATCH --job-name=sleap-infer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=1-$num_files
#SBATCH --partition=ncpu
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

ml purge
ml Anaconda3/2023.09-0
source /camp/apps/eb/software/Anaconda/conda.env.sh

conda activate sleap

FILE_PATH=\$(find \"${DIR}\" -maxdepth 1 -type f -name '*.mp4' | sort | sed -n \"\${SLURM_ARRAY_TASK_ID}p\")

echo \"Processing file \$FILE_PATH\"
echo \"${CENTROID}\"
echo \"${CEN_INS}\"

NAME_VAR=\$(basename \"\$FILE_PATH\" .mp4)
echo $NAME_VAR
echo $FILE_PATH
sleap-track $FILE_PATH -m \"${CENTROID}\" -m \"${CEN_INS}\" -o \"${DIR}\"/\$NAME_VAR.predictions.slp
sleap-convert \"${DIR}\"/\$NAME_VAR.predictions.slp -o \"${DIR}\"/\$NAME_VAR.json --format json
sleap-convert \"${DIR}\"/\$NAME_VAR.predictions.slp -o \"${DIR}\"/\$NAME_VAR.csv --format csv" > sleap-track_array-job.sh

# Submit the dynamically created SBATCH script
sbatch sleap-track_array-job.sh

# remove the SBATCH script
rm sleap-track_array-job.sh