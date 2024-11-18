#!/bin/bash
#SBATCH --job-name=slp-DBSCAN
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --array=1-2
#SBATCH --partition=ncpu
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

ml purge
ml Anaconda3/2023.09-0
ml cuDNN/8.2.1.32-CUDA-11.3.1
source /camp/apps/eb/software/Anaconda/conda.env.sh

conda activate /camp/lab/windingm/home/shared/conda-envs/sleap

# convert ip_string to shell array
IFS=' ' read -r -a path_array <<< "/nemo/lab/windingm/home/shared/sideview-test/2024-07-04_01-30-27_SV27.predictions.feather /nemo/lab/windingm/home/shared/sideview-test/2023-06-29_12-29-14_SV19.predictions.feather"
path_var="${{path_array[$SLURM_ARRAY_TASK_ID-1]}}"
base_var=$(basename "$path_var")

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

echo "Processing mp4: $base_var"
echo "Full path to mp4: $path_var"
echo "Output path: {videos_path}/$base_var.predictions.slp"

cmd="python -u /camp/lab/windingm/home/shared/Crick-HPC-files/sbatch-files/sleap-track_batch-DBSCAN.py -f "$path_var" -e "45" -c "0.9"" 
eval $cmd > python-output_DBSCAN-$base_var.log 2>&1