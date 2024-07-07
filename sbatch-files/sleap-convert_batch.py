import os
import argparse
import tempfile
import subprocess
import time
import json
import csv
import h5py
import pyarrow.feather as feather
import pandas as pd
import numpy as np

# pulling user-input variables from command line
parser = argparse.ArgumentParser(description='sleap_inference: batch inference using sleap models on NEMO')
parser.add_argument('-m', '--model', dest='model', action='store', type=str, required=True, help='type of model')
parser.add_argument('-p', '--videos-path', dest='videos_path', action='store', type=str, default=None, help='path to ip_address list')
parser.add_argument('-j', '--job', dest='job', action='store', type=str, default=None, help='p=predict animal locations, t=track animals, c=convert output file to feather')
parser.add_argument('-f', '--frames', dest='frames', action='store', type=str, default='all', help='track animals?')

# ingesting user-input arguments
args = parser.parse_args()
model = args.model
videos_path = args.videos_path
job = args.job
frames = args.frames

# determine if all frames should be processed or just some
if frames=='all': frame_input = ''
else: frame_input = f' --frames {frames}'

if model == 'sideview': 
    skel_parts = ['head', 'mouthhooks', 'body', 'tail', 'spiracle']

if model == 'topdown': 
    skel_parts = ['head', 'body', 'tail']

if model == 'pupae': 
    skel_parts = ['head', 'body', 'tail']

# identify paths and filenames of all .mp4s in folder
if(os.path.isdir(videos_path)):
    video_file_paths = [f'{videos_path}/{f}' for f in os.listdir(videos_path) if os.path.isfile(os.path.join(videos_path, f)) and (f.endswith('.mp4'))]
    names = [os.path.basename(video_file_path).replace('.mp4', '') for video_file_path in video_file_paths]
else:
    print('Error: -p/--videos-path is not a directory!')

num_videos = len(video_file_paths)

# join all paths together in one string that can be later split by the .sh script
video_file_paths_joined = ' '.join(video_file_paths)
names_joined = ' '.join(names)

# sbatch script to run the array job, to run batch predictions with SLEAP on all videos
convert_script = f"""#!/bin/bash
#SBATCH --job-name=slp-convert
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=1-{num_videos}
#SBATCH --partition=ncpu
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

ml purge
ml Anaconda3/2023.09-0
ml cuDNN/8.2.1.32-CUDA-11.3.1
source /camp/apps/eb/software/Anaconda/conda.env.sh

source activate sleap

# Debugging information
echo "SLURM job id: $SLURM_JOB_ID"
echo "SLURM task id: $SLURM_ARRAY_TASK_ID"
echo "Videos path: {videos_path}"
echo "Model path: {model}"
echo "Running on host: $(hostname)"
echo "Active conda environment: $(conda info --envs | grep \*)"
echo "Installed packages in conda environment:"
conda list

# convert ip_string to shell array
IFS=' ' read -r -a path_array <<< "{video_file_paths_joined}"
path_var="${{path_array[$SLURM_ARRAY_TASK_ID-1]}}"

IFS=' ' read -r -a name_array <<< "{names_joined}"
name_var="${{name_array[$SLURM_ARRAY_TASK_ID-1]}}"

echo "Processing slp: $name_var.predictions.slp"
echo "Full path to slp: {videos_path}.predictions.slp"
echo "Output path: {videos_path}/$name_var.predictions.feather"

cmd="python -u /camp/lab/windingm/home/shared/TestDev/Crick-HPC-files/sbatch-files/sleap-convert_slp.py -p "{videos_path}/$name_var.predictions.slp" -m "{model}"" 
echo $cmd
eval $cmd > python_output_convert-slp.log 2>&1
"""

print(num_videos)
print(video_file_paths_joined)
print(names_joined)
if 'c' in job:
    print('attempting convert job array...')
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_convert_script:
        tmp_convert_script.write(convert_script)
        tmp_script_path = tmp_convert_script.name

    # run the SBATCH script
    process = subprocess.run(["sbatch", tmp_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print('job submitted')
    print(f"stdout: {process.stdout}")
    print(f"stderr: {process.stderr}")
    
    # delete the temporary sbatch file after submission
    os.unlink(tmp_script_path)
