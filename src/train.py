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

def run_training():
    df = pd.read_csv(config.CSV_PATH)
    
    # Split data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])
    
    # Initialize model, optimizer, and loss function
    model = MultimodalEmotionRecognizer(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    
    # --- Phase 1: Class Prototype Learning ---
    print(" Starting Phase 1: Class Prototype Learning...")
    
    # Create dataset for prototype learning (HI intensity samples only)
    prototype_train_dataset = EmotionDataset(train_df, is_train=True, prototype_phase=True)
    prototype_train_loader = DataLoader(prototype_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)

    # Freeze visual network parameters
    for param in model.visual_net.parameters():
        param.requires_grad = False

    # Train for a subset of epochs
    for epoch in range(config.NUM_EPOCHS // 2):
        model.train()
        total_loss = 0
        
        for images, spectrograms, labels in tqdm(prototype_train_loader, desc=f"Phase 1 - Epoch {epoch+1}"):
            images, spectrograms, labels = images.to(config.DEVICE), spectrograms.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images, spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Phase 1 - Epoch {epoch+1} | Train Loss: {total_loss/len(prototype_train_loader):.4f}")

    # --- Phase 2: Supervised Fine-Tuning with full dataset ---
    print(" Starting Phase 2: Supervised Fine-Tuning with full dataset...")
    
    # Unfreeze all model parameters
    for param in model.parameters():
        param.requires_grad = True

    # Re-initialize optimizer for all parameters
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE / 10) # Use a smaller learning rate

    # Create datasets with the full data
    full_train_dataset = EmotionDataset(train_df, is_train=True, prototype_phase=False)
    val_dataset = EmotionDataset(val_df, is_train=False, prototype_phase=False)
    full_train_loader = DataLoader(full_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    for epoch in range(config.NUM_EPOCHS // 2, config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for images, spectrograms, labels in tqdm(full_train_loader, desc=f"Phase 2 - Epoch {epoch+1} [Train]"):
            images, spectrograms, labels = images.to(config.DEVICE), spectrograms.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images, spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # --- Validation ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, spectrograms, labels in tqdm(val_loader, desc=f"Phase 2 - Epoch {epoch+1} [Val]"):
                images, spectrograms, labels = images.to(config.DEVICE), spectrograms.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(images, spectrograms)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(full_train_loader):.4f} | Val Acc: {val_acc:.2f}%")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{config.MODEL_SAVE_PATH}/best_model.pth")
            print(f" Best model saved with accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':
    run_training()