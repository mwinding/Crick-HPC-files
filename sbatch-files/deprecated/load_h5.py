# %%
import h5py

# Function to recursively print the structure of the HDF5 file
def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")

def main():
    # Path to your HDF5 file
    file_path = '/Users/windinm/Desktop/tracks.predictions.000_2024-02-04_16-04-36_SV5.analysis.h5'

    # Open the HDF5 file
    with h5py.File(file_path, 'r') as hdf:
        print("Inspecting the structure of the HDF5 file:")
        # Print the structure of the file
        hdf.visititems(print_structure)
        
        # Example: Access a specific dataset and print a portion of its contents
        dataset_name = 'your_dataset_name'  # Replace with the actual dataset name
        if dataset_name in hdf:
            dataset = hdf[dataset_name]
            print(f"\nContents of the dataset '{dataset_name}':")
            print(dataset[...])  # Print the entire dataset
            # For large datasets, you can print a subset, e.g., the first 10 elements
            # print(dataset[:10])
        else:
            print(f"Dataset '{dataset_name}' not found in the file.")

if __name__ == '__main__':
    main()
# %%
import pandas as pd
import h5py
def process_tracks_h5_to_feather(videos_path, names, skel_parts, track):
    for name in names:
        if track:
            h5_file = f'{videos_path}/{name}.tracks.h5'
            feather_file = f'{videos_path}/{name}.tracks.feather'
        else:
            h5_file = f'{videos_path}/{name}.analysis.h5'
            feather_file = f'{videos_path}/{name}.analysis.feather'

        with h5py.File(h5_file, 'r') as hdf5:
            data = hdf5['tracks'][:].T
            scores = hdf5['tracking_scores'][:]

            # Generate column names based on body parts, identities, and scores
            columns = ['track_id', 'frame']
            for part in skel_parts:
                columns.extend([f'x_{part}', f'y_{part}', f'score_{part}'])

            # Create a list to hold all rows of data
            all_rows = []

            # Loop through each frame
            for frame_idx in range(data.shape[0]):
                # Loop through each identity
                for identity_idx in range(data.shape[3]):
                    row = [identity_idx, frame_idx]
                    for part_idx in range(data.shape[1]):
                        x, y = data[frame_idx, part_idx, :, identity_idx]
                        score = scores[frame_idx, part_idx + 1] if part_idx < len(skel_parts) else np.nan  # Adjust index to skip the first score
                        row.extend([x, y, score])
                    all_rows.append(row)

            # Convert the list to a DataFrame
            df = pd.DataFrame(all_rows, columns=columns)

            # Save the DataFrame to a Feather file
            df.to_feather(feather_file)

videos_path = '/Users/windinm/Desktop'
names = ['tracks.predictions.000_2024-02-04_16-04-36_SV5']
skel_parts = ['head', 'mouthhooks', 'body', 'tail', 'spiracle']
track = False

process_tracks_h5_to_feather(videos_path, names, skel_parts, track)

# %%
# extraction of points!!!
from sleap.io.format import read
file_path = '/Users/windinm/Desktop/slp/2024-02-04_16-04-36_SV5.predictions.slp'
label_obj = read(file_path, for_object='labels')
label_obj.labeled_frames[0]._instances[0].points
# %%
