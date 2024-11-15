# DBSCAN cluster analysis
# adapted from Anna Seggewisse

import joblib
from sklearn.cluster import DBSCAN
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
parser = argparse.ArgumentParser(description='DBSCAN implementation for feather file of SLEAP predictions')
parser.add_argument('-f', '--file_path', dest='file_path', action='store', type=str, required=True, help='file path of feather file')
parser.add_argument('-e', '--eps', dest='eps', action='store', type=int, default=None, help='eps value for DBSCAN')
parser.add_argument('-c', '--cos', dest='cos', action='store', type=float, default=None, help='cosine similarity value for custom DBSCAN')

# ingesting user-input arguments
args = parser.parse_args()
file_path = args.file_path
eps = args.eps
cos = args.cos

# Load the dataset from the Feather file
data = pd.read_feather(file_path)

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

def run_DBSCAN(group, eps, cos):
    data_for_clustering = group[['x_tail', 'y_tail', 'vector_x', 'vector_y']].values
    
    # Apply DBSCAN with the custom metric
    dbscan = DBSCAN(eps=eps, min_samples=3, metric=lambda A, B: custom_distance(A, B, cos))
    labels = dbscan.fit_predict(data_for_clustering)
    
    group['cluster'] = labels
    return group

# Initialize an empty list to store clustering results
clustering_results = []

# Loop through each frame for separate clustering

# add parallelised bit here    
clustering_results = Parallel(n_jobs=-1)(
    delayed(run_DBSCAN)(group, eps=45, cos=0.9) for group in tqdm(coordinates.groupby('frame'), desc="DBSCAN processing .feather files")
)

# Concatenate all frame clustering results into a single DataFrame
result_df = pd.concat(clustering_results, ignore_index=True)

# Save clustering results to a CSV file named after the original Feather file
save_file_path = file_path.replace('.feather', f'.CustomDBSCAN-eps={eps}-cos={cos}.feather')
result_df.to_feather(save_file_path, index=False)
