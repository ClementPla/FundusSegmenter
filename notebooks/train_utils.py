from monai.losses import DiceCELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import BinaryAccuracy
from tqdm.auto import tqdm
import torch
import torch.nn as nn


def train_one_step(images, masks, model, optimizer, criterion: DiceCELoss, device, scheduler):
    """
    Perform one training step: forward pass, loss computation, backward pass, and optimizer step.
    images: input images tensor
    masks: ground truth masks tensor
    model: the neural network model
    optimizer: the optimizer
    criterion: loss function (DiceCELoss)
    device: computation device (CPU or GPU)
    """
    model.train()
    images = images.to(device)
    masks = masks.to(device)

    optimizer.zero_grad()
    outputs = model(images)

    loss = criterion(outputs, masks)
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item()


def train(train_loader, model, device, n_iterations):
    """
    Train the model for one epoch.
    train_loader: DataLoader for training data
    model: the neural network model
    device: computation device (CPU or GPU)
    """
    total_loss = 0.0
    n_batches = 0
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = DiceCELoss(sigmoid=True, label_smoothing=0.3)
    model = model.to(device)
    metric = DiceScore(num_classes=1).to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_iterations)

    for images, masks, style in tqdm(
        train_loader, desc="Training", leave=False, total=n_iterations
    ):
        loss = train_one_step(images, masks, model, optimizer, criterion, device, scheduler)
        total_loss += loss
        n_batches += 1
        if n_batches >= n_iterations:
            break
        if n_batches % 100 == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(images.to(device))
                preds = (outputs.sigmoid() > 0.5).float()
                dice_score = metric(preds, masks.to(device))
            print(
                f"Batch {n_batches}, Loss: {loss:.4f}, Dice Score: {dice_score.item():.4f}"
            )

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return avg_loss



def train_probe(model, dataloader, f_index, device, n_iterations):
    out_channels = model.encoder.out_channels[f_index]
    probe = nn.Linear(out_channels, 1)
    nn.init.xavier_uniform_(probe.weight)
    nn.init.zeros_(probe.bias)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(probe.parameters(), lr=1e-4)
    metric = BinaryAccuracy().to(device)
    probe = probe.to(device)
    model = model.to(device)
    model.eval()
    n_batches = 0
    for images, masks, styles in tqdm(dataloader, desc="Training Probe", leave=False, total=n_iterations):
        images = images.to(device)
        gt = (styles.to(device)>0).float() 
        optimizer.zero_grad()
        with torch.no_grad():
            features = model.encoder(images)[f_index]  # (B, C, H, W)
        B, C, H, W = features.shape
        features = features.mean(dim=(2,3))  # (B, C)
        logits = probe(features).squeeze(1)  # (B)
        loss = criterion(logits, gt)
        loss.backward()
        optimizer.step()
        n_batches += 1
        if n_batches >= n_iterations:
            break
        if n_batches % 100 == 0:
            with torch.no_grad():
                preds = (logits > 0).float()
                acc = metric(preds, gt)
            print(f"Probe Batch {n_batches}, Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")
    return probe