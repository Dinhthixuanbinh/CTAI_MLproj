import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src import config


def evaluate(
    model: nn.Module, dataloader: DataLoader, test: bool = False
) -> tuple[float, float]:
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
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    if test:
        logger.info("Classification Report:")
        logger.info(
            f"\n{classification_report(all_labels, all_preds, zero_division=0)}"
        )

    return acc, f1
