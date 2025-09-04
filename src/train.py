import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
import copy
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src import config
from src.dataset import MultiTaskDataset
from src.model import MultimodalEmotionRecognizer
from src.loss import ContrastiveLoss

def run_training():
    """
    Executes the unified, end-to-end training process for the multi-task model.
    """
    logger.add("training.log", rotation="500 MB")
    
    # Load and split the classification data
    df_classification = pd.read_csv('/kaggle/working/cremad_classification.csv')
    train_ce_df, val_ce_df = train_test_split(df_classification, test_size=0.2, random_state=42, stratify=df_classification['emotion'])

    # Load the contrastive data
    df_contrastive = pd.read_csv('/kaggle/working/cremad_contrastive.csv')

    # Create the unified multi-task datasets
    train_dataset = MultiTaskDataset(train_ce_df, df_contrastive, is_train=True)
    val_dataset = MultiTaskDataset(val_ce_df, df_contrastive, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    logger.info(f"Number of training samples: {len(train_loader.dataset)}")
    logger.info(f"Number of validation samples: {len(val_loader.dataset)}")

    model = MultimodalEmotionRecognizer(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    # Check for multiple GPUs and use DataParallel
    if torch.cuda.device_count() > 1:
        logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # Define loss functions and optimizer
    criterion_ce = nn.CrossEntropyLoss()
    criterion_contrastive = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Tunable loss weights
    w_ce = 1.0
    w_cl = 0.5
    
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(config.NUM_EPOCHS):
        # --- Training Loop ---
        model.train()
        total_loss = 0
        total_ce_loss = 0
        total_cl_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        for images_ce, audio_ce, labels_ce, audio_cl1, audio_cl2 in pbar:
            images_ce = images_ce.to(config.DEVICE)
            audio_ce = audio_ce.to(config.DEVICE)
            labels_ce = labels_ce.to(config.DEVICE)
            audio_cl1 = audio_cl1.to(config.DEVICE)
            audio_cl2 = audio_cl2.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass through the multi-task model
            emotion_logits, audio_emb1, audio_emb2 = model(images_ce, audio_ce, audio_cl1, audio_cl2)

            # Calculate individual losses
            loss_ce = criterion_ce(emotion_logits, labels_ce)
            # The contrastive loss expects `label` to be 0 for similar, 1 for dissimilar.
            # In our case, `audio_cl1` and `audio_cl2` are always dissimilar, so the label is 1.
            loss_cl = criterion_contrastive(audio_emb1, audio_emb2, torch.ones_like(labels_ce).float())
            
            # Combine losses
            total_task_loss = (w_ce * loss_ce) + (w_cl * loss_cl)
            
            total_task_loss.backward()
            optimizer.step()

            total_loss += total_task_loss.item()
            total_ce_loss += loss_ce.item()
            total_cl_loss += loss_cl.item()

            pbar.set_postfix({
                "Total Loss": f"{total_task_loss.item():.4f}",
                "CE Loss": f"{loss_ce.item():.4f}",
                "CL Loss": f"{loss_cl.item():.4f}"
            })

        avg_loss = total_loss / len(train_loader)
        avg_ce_loss = total_ce_loss / len(train_loader)
        avg_cl_loss = total_cl_loss / len(train_loader)
        logger.info(f"End of Epoch {epoch+1} | Train Loss: {avg_loss:.4f} (CE: {avg_ce_loss:.4f}, CL: {avg_cl_loss:.4f})")

        # --- Validation Loop ---
        val_acc, val_f1 = evaluate(model, val_loader)
        logger.info(f"Validation Accuracy: {val_acc:.4f}, F1-score: {val_f1:.4f}")
        
        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            logger.info(f"New best model found at epoch {epoch+1} with validation accuracy: {best_val_acc:.4f}")
            torch.save(best_model_state, f"{config.MODEL_SAVE_PATH}/best_model.pth")
    
    logger.info("Training complete.")
    logger.info(f"Best model saved to {config.MODEL_SAVE_PATH}/best_model.pth")
    
    # --- Final Evaluation on Test Set ---
    logger.info("Running final evaluation with best model...")
    # A proper test set should be used here, for this example we will use the validation set again
    model.load_state_dict(best_model_state)
    test_acc, test_f1 = evaluate(model, val_loader, test=True)
    logger.info(f"Test Accuracy: {test_acc:.4f}, Test F1-score: {test_f1:.4f}")

def evaluate(model: nn.Module, dataloader: DataLoader, test: bool = False) -> tuple[float, float]:
    """
    Evaluates the model on a given dataloader and returns accuracy and F1-score.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for images_ce, audio_ce, labels_ce, _, _ in pbar:
            images_ce = images_ce.to(config.DEVICE)
            audio_ce = audio_ce.to(config.DEVICE)
            labels_ce = labels_ce.to(config.DEVICE)
            
            emotion_logits, _, _ = model(images_ce, audio_ce, audio_ce, audio_ce)
            
            predictions = torch.argmax(emotion_logits, dim=-1)
            
            all_preds.extend(predictions.cpu().tolist())
            all_labels.extend(labels_ce.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    if test:
        logger.info("Classification Report:")
        logger.info(f"\n{classification_report(all_labels, all_preds, zero_division=0)}")
        
    return acc, f1

if __name__ == '__main__':
    run_training()
