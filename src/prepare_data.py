import os
import pandas as pd
from tqdm import tqdm
from itertools import combinations


def create_multi_task_datasets(base_dir: str = "/kaggle/input/copy-crema-d/cremad"):
    """
    Scans the CREMA-D directory structure to create two CSV files:
    1. A standard dataset for classification.
    2. A contrastive dataset with pairs of audio clips from the same
       speaker and sentence, but with different emotions.

    Args:
        base_dir (str): The base directory of the CREMA-D dataset.
    """
    audio_dir = os.path.join(base_dir, "AudioWAV/")
    video_dir = os.path.join(base_dir, "Image-01-FPS/")

    classification_data = []
    contrastive_data = []

    # Create a dictionary to group files by speaker and sentence
    speaker_sentence_groups = {}

    logger.info("Scanning directories to create dataset files...")

    # Iterate through all the audio files to build the classification dataset and grouping for contrastive pairs
    for audio_filename in tqdm(os.listdir(audio_dir), desc="Processing files"):
        if audio_filename.endswith(".wav"):
            parts = audio_filename.split("_")
            # Extract key information from the filename
            file_id = "_".join(parts[0:3]) + "_" + parts[3].split(".")[0]
            speaker_id = parts[0]
            sentence_id = parts[1]
            emotion = parts[2]

            audio_path = os.path.join(audio_dir, audio_filename)
            video_frame_dir = os.path.join(video_dir, file_id)

            # Add to classification data if video frames exist
            if os.path.isdir(video_frame_dir):
                classification_data.append(
                    {
                        "audio_path": audio_path,
                        "video_frame_dir": video_frame_dir,
                        "emotion": emotion,
                        "speaker_id": speaker_id,
                        "sentence_id": sentence_id,
                    }
                )

            # Add to the speaker-sentence-emotion grouping
            key = (speaker_id, sentence_id)
            if key not in speaker_sentence_groups:
                speaker_sentence_groups[key] = {}
            if emotion not in speaker_sentence_groups[key]:
                speaker_sentence_groups[key][emotion] = []
            speaker_sentence_groups[key][emotion].append(audio_path)

    # Now, build the contrastive dataset from the groupings
    for key, emotion_dict in speaker_sentence_groups.items():
        emotions = list(emotion_dict.keys())
        if len(emotions) > 1:
            # Create all possible pairs of clips with different emotions for contrastive learning
            for e1, e2 in combinations(emotions, 2):
                for clip1 in emotion_dict[e1]:
                    for clip2 in emotion_dict[e2]:
                        contrastive_data.append(
                            {
                                "audio_path_1": clip1,
                                "audio_path_2": clip2,
                                "speaker_id": key[0],
                                "sentence_id": key[1],
                            }
                        )

    # Save DataFrames to CSV files
    df_classification = pd.DataFrame(classification_data)
    df_contrastive = pd.DataFrame(contrastive_data)

    classification_save_path = os.path.join(
        "/kaggle/working/", "cremad_classification.csv"
    )
    contrastive_save_path = os.path.join("/kaggle/working/", "cremad_contrastive.csv")

    df_classification.to_csv(classification_save_path, index=False)
    df_contrastive.to_csv(contrastive_save_path, index=False)

    logger.info(
        f"Successfully created classification CSV file with {len(df_classification)} samples."
    )
    logger.info(
        f"Successfully created contrastive CSV file with {len(df_contrastive)} pairs."
    )
    logger.info(f"Saved to: {classification_save_path} and {contrastive_save_path}")


if __name__ == "__main__":
    from loguru import logger

    create_multi_task_datasets()
