import os
import argparse
import pyarrow.feather as feather
import pandas as pd
import numpy as np
from sleap.io.format import read
from joblib import Parallel, delayed

# pulling user-input variables from command line
parser = argparse.ArgumentParser(description='sleap_inference: convert .slp files to .feather on NEMO')
parser.add_argument('-p', '--path', dest='path', action='store', type=str, default=None, help='path to slp file')
parser.add_argument('-m', '--model', dest='model', action='store', type=str, required=True, help='type of model')

# ingesting user-input arguments
args = parser.parse_args()
videos_path = args.path
model = args.model

print('sleap-convert_slp.py started...')

if model == 'sideview': skel_parts = ['head', 'mouthhooks', 'body', 'tail', 'spiracle']
if model == 'topdown': skel_parts = ['head', 'body', 'tail']
if model == 'pupae': skel_parts = ['head', 'body', 'tail']

# convert .slp to .feather
# Process each frame
def process_frame(frame, skel_parts):
    data = []
    for j, instance in enumerate(frame._instances):
        array = [j] + [frame.frame_idx] + list(instance.points_and_scores_array.flatten())
        array = [np.round(x, 2).astype('float32') for x in array]  # reduce size of data
        data.append(array)
    return data

# Convert .slp to .feather
def slp_to_feather(file_path, skel_parts):
    feather_file = file_path.replace('.slp', '.feather')
    label_obj = read(file_path, for_object='labels')

    results = Parallel(n_jobs=-1)(delayed(process_frame)(frame, skel_parts) for frame in label_obj.labeled_frames)

    data = [item for sublist in results for item in sublist]

    columns = ['track_id', 'frame']
    for part in skel_parts:
        columns.extend([f'x_{part}', f'y_{part}', f'score_{part}'])

    df = pd.DataFrame(data, columns=columns)
    df.to_feather(feather_file)

# Identify paths and filenames of all .mp4s in folder
if os.path.isdir(videos_path):
    video_file_paths = [f'{videos_path}/{f}' for f in os.listdir(videos_path) if os.path.isfile(os.path.join(videos_path, f)) and f.endswith('.mp4')]
else:
    print('Error: -p/--videos-path is not a directory!')

video_file_paths = [x.replace('.mp4', '.predictions.slp') for x in video_file_paths]

for path in video_file_paths:
    slp_to_feather(path, skel_parts)

