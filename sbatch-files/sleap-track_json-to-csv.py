# %%
import json
import csv

# import videos_path and names
# import point_ids

videos_path = '/Users/windinm/Desktop/2024-05-01_sleap-test/prediction-video-folder/N10'
names = ['2024-04-15_10-17-45_td4', '2024-04-15_10-28-43_td4']

# convert .json to .csv
for name in names:

    with open(f'{videos_path}/{name}.tracks.json', 'r') as file:
        data = json.load(file)

    data_labels = data['labels']  # Assuming 'data' is the loaded JSON with the 'labels' key

    # Open a CSV file to write to
    with open(f'{videos_path}/{name}.tracks.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row
        writer.writerow(['label_id', 'frame', 'x_head', 'y_head', 'score_head', 'x_body', 'y_body', 'score_body', 'x_tail', 'y_tail', 'score_tail'])
        
        # Loop through each frame in the data
        for frame in data_labels:
            video_id = frame['video']
            frame_idx = frame['frame_idx']

            # Loop through each instance in the frame
            for instance in frame['_instances']:
                # Initialize dictionary to store coordinates and scores
                coords = {
                    'head': {'x': None, 'y': None, 'score': None},
                    'body': {'x': None, 'y': None, 'score': None},
                    'tail': {'x': None, 'y': None, 'score': None}
                }

                # Loop through each point to assign head, body, and tail
                for point_id, point_details in instance['_points'].items():
                    if point_id == '0':  # Assuming 0 is head
                        coords['head'] = {'x': point_details['x'], 'y': point_details['y'], 'score': point_details['score']}
                    elif point_id == '1':  # Assuming 1 is body
                        coords['body'] = {'x': point_details['x'], 'y': point_details['y'], 'score': point_details['score']}
                    elif point_id == '2':  # Assuming 2 is tail
                        coords['tail'] = {'x': point_details['x'], 'y': point_details['y'], 'score': point_details['score']}
                
                # Write row data
                writer.writerow([
                    video_id,
                    frame_idx,
                    coords['head']['x'], coords['head']['y'], coords['head']['score'],
                    coords['body']['x'], coords['body']['y'], coords['body']['score'],
                    coords['tail']['x'], coords['tail']['y'], coords['tail']['score']
                ])

# %%
