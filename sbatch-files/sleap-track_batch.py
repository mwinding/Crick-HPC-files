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
from sleap.io.format import read
from joblib import Parallel, delayed
from tqdm import tqdm

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

elif model == 'topdown': 
    path = '/camp/lab/windingm/home/shared/models/topdown/active/'
    skel_parts = ['head', 'body', 'tail']

elif model == 'pupae': 
    path = '/camp/lab/windingm/home/shared/models/pupae/active/'
    skel_parts = ['head', 'body', 'tail']

else:
    raise ValueError('Model parameter must be "sideview", "topdown", or "pupae"!')
    

centroid_model, centered_model = find_models(path)

# identify paths and filenames of all .mp4s in folder
if(os.path.isdir(videos_path)):
    video_file_paths = [f'{videos_path}/{f}' for f in os.listdir(videos_path) if os.path.isfile(os.path.join(videos_path, f)) and (f.endswith('.mp4')) and not (f.endswith('playback.mp4'))]
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
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

ml purge
ml Anaconda3/2023.09-0
ml cuDNN/8.2.1.32-CUDA-11.3.1
source /camp/apps/eb/software/Anaconda/conda.env.sh

conda activate /camp/lab/windingm/home/shared/conda-envs/sleap

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

if 'p' in job:
    script += f"""
echo "Output path: {videos_path}/$name_var.predictions.h5"
sleap-track $path_var --verbosity rich --batch_size 32{frame_input} -m {centroid_model} -m {centered_model} -o {videos_path}/$name_var.predictions.slp
"""

if 't' in job:
    script += f"""
echo "Output path: {videos_path}/$name_var.tracks.slp"
sleap-track --tracking.tracker simple --verbosity rich -o {videos_path}/$name_var.tracks.slp {videos_path}/$name_var.predictions.slp
"""

#if ('c' in job) and ('t' in job):
#    script += f"""
#sleap-convert {videos_path}/$name_var.tracks.slp -o {videos_path}/$name_var.tracks.h5 --format analysis
#"""

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

if ('p' in job) or ('t' in job):
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
    
    check_job_completed(job_id)

# convert .slp to .feather
def slp_to_feather(file_path, skel_parts, track_ids):

    feather_file = file_path.replace('.slp', '.feather')
    label_obj = read(file_path, for_object='labels')

    data = []
    for i, frame in enumerate(label_obj.labeled_frames):
        for j, instance in enumerate(frame._instances):
            if track_ids: j = int(instance.track.name.replace("track_", "")) # adds in the track ID instead of the jth index
            array = [j] + [i] + [instance.score] + list(instance.points_and_scores_array.flatten())
            array = [np.round(x, 2).astype('float32') for x in array] # reduce size of data
            data.append(array)

    columns = ['track_id', 'frame', 'instance_score']
    for part in skel_parts:
        columns.extend([f'x_{part}', f'y_{part}', f'score_{part}'])

    df = pd.DataFrame(data, columns = columns)
    df.reset_index(drop=True, inplace=True) # prevents a serialisation error in feather conversion
    df.to_feather(feather_file)

# convert all .mp4s names to .slp names folder
if 't' in job: video_file_paths = [x.replace('.mp4', '.tracks.slp') for x in video_file_paths]
else: video_file_paths = [x.replace('.mp4', '.predictions.slp') for x in video_file_paths]

# Parallelize the conversion of .slp files to .feather files
if 'c' in job:
    if 't' in job: track_ids = True
    if 't' not in job: track_ids = False

    print(video_file_paths)

    Parallel(n_jobs=-1)(
        delayed(slp_to_feather)(path, skel_parts, track_ids=track_ids) for path in tqdm(video_file_paths, desc="Processing .slp files")
    )

# DBSCAN cluster analysis
# adapted from Anna Seggewisse
    
eps = 45
cos = 0.9

def DBSCAN_cluster(file_path, eps, cos):

    # Load the dataset from the Feather file
    data = pd.read_feather(file_path)

    # Filter data for every 10th frame
    #filtered_data = data[(data['frame'] % 10 == 0)]

    # Select the relevant columns for DBSCAN (head and tail coordinates)
    coordinates = data[['track_id', 'frame', 'x_head', 'y_head', 'x_tail', 'y_tail']].dropna()

    # Calculate vectors for each instance (tail - head)
    coordinates['vector_x'] = coordinates['x_tail'] - coordinates['x_head']
    coordinates['vector_y'] = coordinates['y_tail'] - coordinates['y_head']

    # Define the custom distance function with separate checks
    def custom_distance(A, B, cos):
        tail_A = A[:2]  # x_tail and y_tail of A
        tail_B = B[:2]  # x_tail and y_tail of B
        euclidean_tail_dist = np.linalg.norm(tail_A - tail_B)
        
        vector_A = A[2:]  # vector_x and vector_y of A
        vector_B = B[2:]  # vector_x and vector_y of B
        magnitude_A = np.linalg.norm(vector_A)
        magnitude_B = np.linalg.norm(vector_B)
        
        if magnitude_A == 0 or magnitude_B == 0:
            return 1000
        
        cos_similarity = np.dot(vector_A, vector_B) / (magnitude_A * magnitude_B)
        
        if cos_similarity > cos:
            return euclidean_tail_dist
        
        return 1000

    # Initialize an empty list to store clustering results
    clustering_results = []

    # Loop through each frame for separate clustering
    for frame, group in coordinates.groupby('frame'):
        data_for_clustering = group[['x_tail', 'y_tail', 'vector_x', 'vector_y']].values
        
        # Apply DBSCAN with the custom metric
        dbscan = DBSCAN(eps=eps, min_samples=3, metric=lambda A, B: custom_distance(A, B, cos))
        labels = dbscan.fit_predict(data_for_clustering)
        
        group['cluster'] = labels
        clustering_results.append(group)

    # Concatenate all frame clustering results into a single DataFrame
    result_df = pd.concat(clustering_results, ignore_index=True)

    # Save clustering results to a CSV file named after the original Feather file
    save_file_path = f"{file_path.replace('.feather', '')}_CustomDBSCANeps={eps}-cos={cos}.csv"
    result_df.to_csv(save_file_path, index=False)

if 'd' in job:
    feather_file_paths = [x.replace('.slp', '.feather') for x in video_file_paths]

    print(feather_file_paths)

    Parallel(n_jobs=-1)(
        delayed(DBSCAN_cluster)(path, eps=45, cos=0.9) for path in tqdm(feather_file_paths, desc="DBSCAN processing .feather files")
    )