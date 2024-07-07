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

# pulling user-input variables from command line
parser = argparse.ArgumentParser(description='sleap_inference: convert .slp files to .feather on NEMO')
parser.add_argument('-p', '--path', dest='path', action='store', type=str, default=None, help='path to slp file')
parser.add_argument('-m', '--model', dest='model', action='store', type=str, required=True, help='type of model')

# ingesting user-input arguments
args = parser.parse_args()
path = args.path
model = args.model

print('sleap-convert_slp.py started...')

if model == 'sideview': skel_parts = ['head', 'mouthhooks', 'body', 'tail', 'spiracle']
if model == 'topdown': skel_parts = ['head', 'body', 'tail']
if model == 'pupae': skel_parts = ['head', 'body', 'tail']

# convert .slp to .feather
def slp_to_feather(file_path, skel_parts):

    feather_file = file_path.replace('.slp', '.feather')
    label_obj = read(file_path, for_object='labels')

    data = []
    for i, frame in enumerate(label_obj.labeled_frames):
        for j, instance in enumerate(frame._instances):
            array = [j] + [i] + list(instance.points_and_scores_array.flatten())
            array = [np.round(x, 2).astype('float32') for x in array] # reduce size of data
            data.append(array)

    columns = ['track_id', 'frame']
    for part in skel_parts:
        columns.extend([f'x_{part}', f'y_{part}', f'score_{part}'])

    df = pd.DataFrame(data, columns = columns)
    df.to_feather(feather_file)

slp_to_feather(path, skel_parts)