# src/prepare_data.py

import os
import pandas as pd
from tqdm import tqdm

def create_dataset_csv(base_dir='../cremad/'):
    """
    Scans the CREMA-D directory structure to create a CSV file with paths
    to audio files, corresponding video frame directories, and emotion labels.
    """
    audio_dir = os.path.join(base_dir, 'AudioWAV/')
    video_dir = os.path.join(base_dir, 'Image-01-FPS/')
    
    data = []
    
    print("Scanning directories to create dataset file...")
    # Iterate through all the audio files
    for audio_filename in tqdm(os.listdir(audio_dir)):
        if audio_filename.endswith('.wav'):
            # Extract the unique identifier and emotion from the filename
            parts = audio_filename.split('_')
            file_id = '_'.join(parts[0:3]) + '_' + parts[3].split('.')[0] # e.g., 1001_DFA_ANG_XX
            emotion = parts[2] # e.g., ANG

            # Construct the full paths
            audio_path = os.path.join(audio_dir, audio_filename)
            video_frame_dir = os.path.join(video_dir, file_id)
            
            # Ensure the corresponding video frame directory exists
            if os.path.isdir(video_frame_dir):
                data.append({
                    'audio_path': audio_path,
                    'video_frame_dir': video_frame_dir,
                    'emotion': emotion
                })

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(data)
    save_path = os.path.join('../data/', 'cremad_paths.csv')
    df.to_csv(save_path, index=False)
    
    print(f"Successfully created CSV file with {len(df)} samples.")
    print(f"Saved to: {save_path}")

if __name__ == '__main__':
    create_dataset_csv()