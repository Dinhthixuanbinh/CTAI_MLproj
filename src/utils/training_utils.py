import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch.nn as nn

def train_epoch(config, epoch, model, device, dataloader, optimizer, scheduler):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for spec, image, label in tqdm(dataloader, desc=f'Epoch {epoch}'):
        spec, image, label = spec.to(device), image.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(spec.float(), image.float())
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
    scheduler.step()

def eval_model(config, model, device, dataloader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    golds, preds = [], []
    total_loss = 0
    with torch.no_grad():
        for spec, image, label in dataloader:
            spec, image, label = spec.to(device), image.to(device), label.to(device)
            out = model(spec.float(), image.float())
            loss = criterion(out, label)
            total_loss += loss.item()
            y_hat = torch.argmax(out, dim=-1)
            golds.extend(label.cpu().numpy())
            preds.extend(y_hat.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    weighted_f1 = f1_score(golds, preds, average='weighted')
    return avg_loss, weighted_f1