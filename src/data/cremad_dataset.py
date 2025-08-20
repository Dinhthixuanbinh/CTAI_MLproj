import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from src.data.data_preprocessing.visual_preprocessing import VisualProcessor
from src.data.data_preprocessing.audio_preprocessing import AudioProcessor

class CremadDataset(Dataset):
    def __init__(self, config, data, train=True):
        self.config = config
        self.data = data
        self.train = train
        self.visual_processor = VisualProcessor(config, train)
        self.audio_processor = AudioProcessor()
        self.class_dict = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        file_name, label_name = item[0], item[1]

        # Process audio and visual data
        audio_spec = self.audio_processor.process_audio(os.path.join(self.config['audio_path'], f"{file_name}.wav"))
        visual_frames = self.visual_processor.process_visual(os.path.join(self.config['visual_path'], file_name))
        
        label = self.class_dict[label_name]
        
        return audio_spec, visual_frames, label

def load_cremad(config, data_root='./data'):
    train_csv = os.path.join(data_root, config['dataset'], 'train.csv')
    test_csv = os.path.join(data_root, config['dataset'], 'test.csv')

    train_df = pd.read_csv(train_csv, header=None)
    train, dev = train_test_split(train_df, test_size=0.1)
    test = pd.read_csv(test_csv, header=None)

    train_dataset = CremadDataset(config, train.to_numpy(), train=True)
    dev_dataset = CremadDataset(config, dev.to_numpy(), train=False)
    test_dataset = CremadDataset(config, test.to_numpy(), train=False)

    return train_dataset, dev_dataset, test_dataset