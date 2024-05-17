import os
import argparse
import tempfile
import subprocess

# pulling user-input variables from command line
parser = argparse.ArgumentParser(description='sleap_inference: batch inference using sleap models on NEMO')
parser.add_argument('-m1', '--centroid-model', dest='centroid_model', action='store', type=str, required=True, help='path to centroid model')
parser.add_argument('-m2', '--centered-model', dest='centered_model', action='store', type=str, required=True, help='path to centered instance model')
parser.add_argument('-p', '--videos-path', dest='videos_path', action='store', type=str, default=None, help='path to ip_address list')

# ingesting user-input arguments
args = parser.parse_args()
centroid_model = args.centroid_model
centered_model = args.centered_model
videos_path = args.videos_path

# batch process videos in folder
if(os.path.isdir(videos_path)):
    video_file_paths = [f'{videos_path}/{f}' for f in os.listdir(videos_path) if os.path.isfile(os.path.join(videos_path, f)) and (f.endswith('.mp4'))]
    names = [os.path.basename(video_file_path).replace('.mp4', '') for video_file_path in video_file_paths]
else:
    print('Error: -p/--videos-path is not a directory!')

# identify number of videos
num_videos = len(video_file_paths)
video_file_paths_joined = ' '.join(video_file_paths)
names_joined = ' '.join(names)

# sbatch script to run the array job
script = f"""#!/bin/bash
#SBATCH --job-name=sleap-infer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=1-{num_videos}
#SBATCH --partition=ncpu
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

ml purge
ml Anaconda3/2023.09-0
source /camp/apps/eb/software/Anaconda/conda.env.sh

conda activate sleap

# convert ip_string to shell array
IFS=' ' read -r -a path_array <<< "{video_file_paths_joined}"
path_var="${{path_array[$SLURM_ARRAY_TASK_ID-1]}}"

IFS=' ' read -r -a name_array <<< "{names_joined}"
name_var="${{name_array[$SLURM_ARRAY_TASK_ID-1]}}"

echo "Processing mp4: $name_var"
echo "Full path to mp4: $path_var"
echo "Centroid model path: {centroid_model}"
echo "Centered instance model path: {centered_model}"
echo "Output path: {videos_path}/$name_var.predictions.slp"
echo "Output path: {videos_path}/$name_var.tracks.slp"
echo "Output path: {videos_path}/$name_var.tracks.json"

sleap-track $path_var -m {centroid_model} -m {centered_model} -o {videos_path}/$name_var.predictions.slp
sleap-track --tracking.tracker flow -o {videos_path}/$name_var.tracks.slp {videos_path}/$name_var.predictions.slp
sleap-convert {videos_path}/$name_var.tracks.slp -o {videos_path}/$name_var.tracks.json --format json
"""

# Create a temporary file to hold the SBATCH script
with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_script:
    tmp_script.write(script)
    tmp_script_path = tmp_script.name

# Submit the SBATCH script
process = subprocess.run(["sbatch", tmp_script_path])

# Delete the temporary file after submission
os.unlink(tmp_script_path)