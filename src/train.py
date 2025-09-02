# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.dataset import EmotionDataset
from src.model import MultimodalEmotionRecognizer

# Assuming ContrastiveLoss is in a separate file or defined here for simplicity.
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

def run_training():
    df = pd.read_csv(config.CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])
    
    model = MultimodalEmotionRecognizer(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    best_val_acc = 0.0

    # --- PHASE 1: PROTOTYPE LEARNING (CLASSIFICATION) ---
    print("Starting Phase 1: Prototype Learning...")
    
    criterion_ce = nn.CrossEntropyLoss()
    optimizer_ce = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Use a dataset that loads a single image and audio
    phase1_train_dataset = EmotionDataset(train_df, is_train=True, phase2_mode=False)
    phase1_train_loader = DataLoader(phase1_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)

    for epoch in range(config.NUM_EPOCHS // 2):
        model.train()
        total_loss = 0
        for images, spectrograms, labels in tqdm(phase1_train_loader, desc=f"Phase 1 - Epoch {epoch+1}"):
            images, spectrograms, labels = images.to(config.DEVICE), spectrograms.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer_ce.zero_grad()
            outputs = model(images, spectrograms)
            loss = criterion_ce(outputs, labels)
            
            loss.backward()
            optimizer_ce.step()
            total_loss += loss.item()
        
        print(f"Phase 1 - Epoch {epoch+1} | Train Loss: {total_loss/len(phase1_train_loader):.4f}")
    
    # --- PHASE 2: CONTRASTIVE LEARNING ---
    print("Starting Phase 2: Contrastive Learning...")

    # Freeze the visual network
    for param in model.visual_net.parameters():
        param.requires_grad = False
        
    criterion_contrastive = ContrastiveLoss() 
    # Use a new optimizer for the unfrozen parameters (audio_net and fusion layer)
    optimizer_contrastive = optim.Adam(model.parameters(), lr=config.LEARNING_RATE / 10)

    # Note: The dataset and dataloader below are conceptual.
    # A custom dataset class would be needed to return pairs of audio samples and a label
    # indicating if they are from the same speaker/speech.
    # For this example, we will simulate the data.
    
    # Simulating a contrastive dataset for demonstration purposes
    simulated_pairs = [
        (torch.randn(1, 1, config.N_MELS, config.AUDIO_TIME_STEPS), torch.randn(1, 1, config.N_MELS, config.AUDIO_TIME_STEPS), torch.tensor([0])),
        (torch.randn(1, 1, config.N_MELS, config.AUDIO_TIME_STEPS), torch.randn(1, 1, config.N_MELS, config.AUDIO_TIME_STEPS), torch.tensor([1]))
    ]
    
    for epoch in range(config.NUM_EPOCHS // 2, config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        # In a real implementation, you would iterate over a DataLoader
        for audio1_spec, audio2_spec, labels in simulated_pairs: 
            # Note: The model forward pass needs to be modified to handle this.
            # Here we are just using the audio network.
            
            optimizer_contrastive.zero_grad()
            
            audio_enc1 = model.audio_net(audio1_spec.to(config.DEVICE))
            audio_enc2 = model.audio_net(audio2_spec.to(config.DEVICE))
            
            loss = criterion_contrastive(audio_enc1, audio_enc2, labels.to(config.DEVICE))
            loss.backward()
            optimizer_contrastive.step()
            total_loss += loss.item()
        
        print(f"Phase 2 - Epoch {epoch+1} | Contrastive Loss (Simulated): {total_loss / len(simulated_pairs):.4f}")

    # --- FINAL EVALUATION ---
    print("Starting final evaluation...")
    
    val_dataset = EmotionDataset(val_df, is_train=False, phase2_mode=False)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, spectrograms, labels in tqdm(val_loader, desc="Final Evaluation"):
            images, spectrograms, labels = images.to(config.DEVICE), spectrograms.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images, spectrograms)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_acc = 100 * correct / total
    print(f"Final Accuracy: {final_acc:.2f}%")
    
    # Save the final model
    torch.save(model.state_dict(), f"{config.MODEL_SAVE_PATH}/final_model.pth")
    print(f"Final model saved to: {config.MODEL_SAVE_PATH}/final_model.pth")

if __name__ == '__main__':
    run_training()