import os
import argparse
import tempfile
import subprocess
import time
import json
import csv
#import h5py
#import pyarrow.feather as feather

# pulling user-input variables from command line
parser = argparse.ArgumentParser(description='sleap_inference: batch inference using sleap models on NEMO')
parser.add_argument('-m', '--model', dest='model', action='store', type=str, required=True, help='type of model')
parser.add_argument('-p', '--videos-path', dest='videos_path', action='store', type=str, default=None, help='path to ip_address list')
parser.add_argument('-t', '--track', dest='track', action='store', type=str, default=None, help='track animals?')
parser.add_argument('-f', '--frames', dest='frames', action='store', type=str, default='all', help='track animals?')

# ingesting user-input arguments
args = parser.parse_args()
model = args.model
videos_path = args.videos_path
track = args.track
frames = args.frames

track = track.lower() == 'true' # convert to boolean, accepting 'True' or 'true' as input

# determine if all frames should be processed or just some
if frames=='all': frame_input = ''
else: frame_input = f' --frames {frames}'

# identify and set model paths
def find_models(path):
    centroid_model = []
    centered_model = []

    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            if '.centroid' in dir_name:
                full_path = full_path + '/training_config.json'
                centroid_model.append(full_path)
            elif '.centered_instance' in dir_name:
                full_path = full_path + '/training_config.json'
                centered_model.append(full_path)

    if(len(centroid_model)>1 and len(centered_model)>1): raise Exception(f"Multiple centroid and centered models detected! \nInvestigate in this directory: \n{path}")
    if(len(centroid_model)>1): raise Exception(f"Multiple centroid models detected! \nInvestigate in this directory: \n{path}")
    if(len(centered_model)>1): raise Exception(f"Multiple centered models detected! \nInvestigate in this directory: \n{path}")

    return(centroid_model[0], centered_model[0])

if model == 'sideview': 
    path = '/camp/lab/windingm/home/shared/models/sideview/active/'
    skel_parts = ['head', 'mouthhooks', 'body', 'tail', 'spiracle']

if model == 'topdown': 
    path = '/camp/lab/windingm/home/shared/models/topdown/active/'
    skel_parts = ['head', 'body', 'tail']

if model == 'pupae': 
    path = '/camp/lab/windingm/home/shared/models/pupae/active/'
    skel_parts = ['head', 'body', 'tail']

centroid_model, centered_model = find_models(path)

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
script = f"""#!/bin/bash
#SBATCH --job-name=slp-infer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=1-{num_videos}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

ml purge
ml Anaconda3/2023.09-0
ml cuDNN/8.2.1.32-CUDA-11.3.1
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
"""

if track: # if the user wants to do tracking
    script += f"""
echo "Output path: {videos_path}/$name_var.tracks.slp"
echo "Output path: {videos_path}/$name_var.tracks.h5"

sleap-track $path_var --verbosity rich{frame_input} -m {centroid_model} -m {centered_model} -o {videos_path}/$name_var.predictions.slp
sleap-track --tracking.tracker simple --verbosity rich -o {videos_path}/$name_var.tracks.slp {videos_path}/$name_var.predictions.slp
sleap-convert {videos_path}/$name_var.tracks.slp -o {videos_path}/$name_var.tracks.csv --format analysis
"""
else: # if the user doesn't want to do tracking
    script += f"""
echo "Output path: {videos_path}/$name_var.predictions.h5"

sleap-track $path_var --verbosity rich{frame_input} -m {centroid_model} -m {centered_model} -o {videos_path}/$name_var.predictions.slp
sleap-convert {videos_path}/$name_var.predictions.slp -o {videos_path}/$name_var.predictions.csv --format analysis
"""

# Create a temporary file to hold the SBATCH script
with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_script:
    tmp_script.write(script)
    tmp_script_path = tmp_script.name

# run the SBATCH script
process = subprocess.run(["sbatch", tmp_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# delete the temporary sbatch file after submission
os.unlink(tmp_script_path)

# Check the result and extract job ID from the output
if process.returncode == 0:
    job_id_output = process.stdout.strip()
    print(f'\t{job_id_output}')

    job_id = job_id_output.split()[-1]

# wait until all jobs are done
def check_job_completed(job_id, initial_wait=120, wait=120):
        seconds = initial_wait
        print(f"\tWait for {seconds} seconds before checking if slurm job has completed")
        time.sleep(seconds)
        
        # Wait for the array job to complete
        print(f"\tWaiting for slurm job {job_id} to complete...")
        while not is_job_completed(job_id):
            print(f"\tSlurm job {job_id} is still running. Waiting...")
            time.sleep(wait)  # Check every 30 seconds

        print(f"\tSlurm job {job_id} has completed.\n")

def is_job_completed(job_id):
    cmd = ["sacct", "-j", f"{job_id}", "--format=JobID,State", "--noheader"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    lines = result.stdout.strip().split('\n')

    # Initialize flags
    all_completed = True

    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue  # Skip any malformed lines

        job_id_part, job_state = parts[0], parts[1]

        # Check for the main job ID and any array tasks
        if job_id_part == job_id or "_" in job_id_part:  # This line is modified to also consider the main job
            if job_state not in ["COMPLETED", "FAILED", "CANCELLED"]:
                all_completed = False
                break

    return all_completed

check_job_completed(job_id)
'''
# convert tracking JSONs to CSVs
def process_tracks_h5(videos_path, names, skel_parts, track):
    for name in names:
        if track:
            h5_file = f'{videos_path}/{name}.tracks.h5'
            csv_file = f'{videos_path}/{name}.tracks.csv'
        else:
            h5_file = f'{videos_path}/{name}.predictions.h5'
            csv_file = f'{videos_path}/{name}.predictions.csv'

        with h5py.File(hdf5_file, 'r') as hdf5:
            data = hdf5['tracks'][:].T

            # Generate column names based on body parts
            columns = ['track_id', 'frame'] + [f'{coord}_{part}' for part in skel_parts for coord in ['x', 'y', 'score']]

            df = pd.DataFrame(data, columns=columns)

            
            # # Open a CSV file to write to
            # with open(csv_file, mode='w', newline='') as file:
            #     writer = csv.writer(file)
                
            #     # Write the header row
            #     writer.writerow(columns)
                
            #     # Loop through each frame in the data
            #     for frame in data_labels:
            #         video_id = frame['video']
            #         frame_idx = frame['frame_idx']

            #         # Loop through each instance in the frame
            #         for instance in frame['_instances']:
            #             track_id = instance['track']

            #             # Initialize dictionary to store coordinates and scores
            #             coords = {part: {'x': None, 'y': None, 'score': None} for part in skel_parts}

            #             # Loop through each point to assign coordinates and scores
            #             for point_id, point_details in instance['_points'].items():
            #                 part_name = skel_parts[int(point_id)]
            #                 coords[part_name] = {'x': point_details['x'], 'y': point_details['y'], 'score': point_details['score']}
                        
            #             # Write row data
            #             row = [track_id, frame_idx]
            #             for part in skel_parts:
            #                 row.extend([coords[part]['x'], coords[part]['y'], coords[part]['score']])
            #             writer.writerow(row)
            

process_tracks_json(videos_path, names, skel_parts, track)
'''