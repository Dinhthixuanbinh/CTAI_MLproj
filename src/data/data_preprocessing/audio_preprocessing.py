import librosa
import numpy as np
import torch

class AudioProcessor:
    def __init__(self):
        pass

    def process_audio(self, audio_path):
        samples, rate = librosa.load(audio_path, sr=22050)
        resamples = np.tile(samples, 3)[:22050 * 3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        return torch.from_numpy(spectrogram)