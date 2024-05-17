import os
import argparse
import tempfile
import subprocess
import time
import json
import csv

# pulling user-input variables from command line
parser = argparse.ArgumentParser(description='sleap_inference: batch inference using sleap models on NEMO')
parser.add_argument('-m1', '--centroid-model', dest='centroid_model', action='store', type=str, required=True, help='path to centroid model')
parser.add_argument('-m2', '--centered-model', dest='centered_model', action='store', type=str, required=True, help='path to centered instance model')
parser.add_argument('-p', '--videos-path', dest='videos_path', action='store', type=str, default=None, help='path to ip_address list')
parser.add_argument('-s', '--skeleton-parts', dest='skel_parts', action='store', type=str, nargs='+', default=None, help='all node names in SLEAP skeleton')

# ingesting user-input arguments
args = parser.parse_args()
centroid_model = args.centroid_model
centered_model = args.centered_model
videos_path = args.videos_path
skel_parts = args.skel_parts
print(skel_parts)

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

# run the SBATCH script
job_id = subprocess.run(["sbatch", tmp_script_path])

# delete the temporary sbatch file after submission
os.unlink(tmp_script_path)

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

# convert tracking JSONs to CSVs
def process_tracks_json(videos_path, names, skel_parts):
    for name in names:
        with open(f'{videos_path}/{name}.tracks.json', 'r') as file:
            data = json.load(file)

        data_labels = data['labels']

        # Generate column names based on body parts
        columns = ['label_id', 'frame'] + [f'{coord}_{part}' for part in skel_parts for coord in ['x', 'y', 'score']]
        
        # Open a CSV file to write to
        with open(f'{videos_path}/{name}.tracks.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header row
            writer.writerow(columns)
            
            # Loop through each frame in the data
            for frame in data_labels:
                video_id = frame['video']
                frame_idx = frame['frame_idx']

                # Loop through each instance in the frame
                for instance in frame['_instances']:
                    # Initialize dictionary to store coordinates and scores
                    coords = {part: {'x': None, 'y': None, 'score': None} for part in skel_parts}

                    # Loop through each point to assign coordinates and scores
                    for point_id, point_details in instance['_points'].items():
                        part_name = skel_parts[int(point_id)]
                        coords[part_name] = {'x': point_details['x'], 'y': point_details['y'], 'score': point_details['score']}
                    
                    # Write row data
                    row = [video_id, frame_idx]
                    for part in skel_parts:
                        row.extend([coords[part]['x'], coords[part]['y'], coords[part]['score']])
                    writer.writerow(row)

process_tracks_json(videos_path, names, skel_parts)