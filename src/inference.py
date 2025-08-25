# src/inference.py

import torch
import cv2
import torchaudio
import numpy as np
from src import config
from src.model import MultimodalEmotionRecognizer
from src.dataset import EmotionDataset # Re-use preprocessing logic

def predict_emotion(video_path, audio_path, model_path):
    # Load the trained model
    model = MultimodalEmotionRecognizer(num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    # Create a dummy dataframe to reuse the Dataset's preprocessing
    dummy_df = pd.DataFrame([{'video_path': video_path, 'audio_path': audio_path, 'emotion': 'NEU'}])
    inference_dataset = EmotionDataset(dummy_df, is_train=False)
    
    # Get preprocessed data for a single item
    image, spectrogram, _ = inference_dataset[0]
    
    # Add a batch dimension and send to device
    image = image.unsqueeze(0).to(config.DEVICE)
    spectrogram = spectrogram.unsqueeze(0).to(config.DEVICE)

    # Make prediction
    with torch.no_grad():
        outputs = model(image, spectrogram)
        _, predicted_idx = torch.max(outputs, 1)

    # Map index back to emotion label
    idx_to_emotion = {v: k for k, v in config.EMOTION_MAP.items()}
    predicted_emotion = idx_to_emotion[predicted_idx.item()]
    
    print(f"Predicted Emotion: {predicted_emotion}")
    return predicted_emotion

if __name__ == '__main__':
    # --- Example Usage ---
    VIDEO_FILE = 'path/to/your/video.mp4'
    AUDIO_FILE = 'path/to/your/audio.wav'
    MODEL_FILE = '../models/best_model.pth'

    predict_emotion(VIDEO_FILE, AUDIO_FILE, MODEL_FILE)