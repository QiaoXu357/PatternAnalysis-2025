import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

from .modules import CustomConvNeXt


class ModelEMA:
    """Exponential Moving Average (EMA) of model parameters.

    Keeps a non-trainable copy of the model updated as ema = d*ema + (1-d)*model.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = self._clone_model(model)
        self.ema.eval()
        self.decay = decay

    @torch.no_grad()
    def _clone_model(self, model):
        ema = type(model)().to(next(model.parameters()).device)
        ema.load_state_dict(model.state_dict())
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(v * d + msd[k] * (1.0 - d))

def train_epoch(model, loader, criterion, optimizer, device, grad_clip: float | None = 1.0, ema: ModelEMA | None = None):
    """Run one training epoch and optionally update EMA."""
    model.train()
    running_loss = 0.0
    predictions, targets = [], []
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())
    epoch_loss = running_loss / max(1, len(loader))
    epoch_acc = accuracy_score(targets, predictions) if predictions else 0.0
    return epoch_loss, epoch_acc


def _predict_with_tta(model, images, tta: int = 1):
    """Apply lightweight test-time augmentation (horizontal flip) if enabled."""
    logits = model(images)
    if tta <= 1:
        return logits
    outs = [logits]
    # Simple TTA: Horizontal Flip
    images_flip = torch.flip(images, dims=[3])
    outs.append(model(images_flip))
    # More TTAs (such as different scales) can be added, but keep it lightweight here.
    return torch.stack(outs, dim=0).mean(0)


def evaluate(model, loader, criterion, device, tta: int = 1):
    """Evaluate model over a DataLoader and compute loss/accuracy."""
    model.eval()
    running_loss = 0.0
    predictions, targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = _predict_with_tta(model, images, tta=tta)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    epoch_loss = running_loss / max(1, len(loader))
    epoch_acc = accuracy_score(targets, predictions) if predictions else 0.0
    return epoch_loss, epoch_acc, predictions, targets


def plot_training_history(train_losses, val_losses, train_accs, val_accs, out_path: str):
    """Plot loss and accuracy curves and save to a file."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend(); ax1.grid(True)
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_confusion_matrix(cm, labels, out_path: str):
    """Render and save a confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train(
    data_loaders,
    num_epochs: int = 50,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    label_smoothing: float = 0.1,
    warmup_epochs: int = 5,
    ema_decay: float = 0.999,
    tta: int = 2,
    model_kwargs: dict | None = None,
    device: torch.device | None = None,
    output_dir: str = ".",
):
    """Train the model end-to-end and report final test metrics.

    Uses linear warmup followed by cosine LR schedule, label smoothing,
    gradient clipping, EMA validation, and optional TTA at test time.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_loader, val_loader, test_loader = data_loaders
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomConvNeXt(**(model_kwargs or {})).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # warmup + cosine
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_epochs))
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs))
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[max(1, warmup_epochs)])

    ema = ModelEMA(model, decay=ema_decay)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0
    best_path = os.path.join(output_dir, 'best_model.pth')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=1.0, ema=ema)
        # Use EMA weights for verification
        va_loss, va_acc, _, _ = evaluate(ema.ema, val_loader, criterion, device, tta=1)
        scheduler.step()
        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_accs.append(tr_acc); val_accs.append(va_acc)
        print(f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}")
        print(f"Val Loss: {va_loss:.4f}, Val Acc: {va_acc:.4f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model with Val Acc: {va_acc:.4f} -> {best_path}")

    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        out_path=os.path.join(output_dir, 'training_history.png'),
    )

    model.load_state_dict(torch.load(best_path, map_location=device))
    # Enable TTA (Simple Horizontal Flip) during the testing phase
    test_loss, test_acc, test_preds, test_targets = evaluate(model, test_loader, criterion, device, tta=tta)
    print(f"\n{'=' * 50}")
    print(f"FINAL TEST ACCURACY: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"{'=' * 50}")
    cm = confusion_matrix(test_targets, test_preds)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n",
          classification_report(test_targets, test_preds, target_names=['Normal', 'AD']))
    plot_confusion_matrix(cm, ['Normal', 'AD'], out_path=os.path.join(output_dir, 'confusion_matrix.png'))
    return {
        'model': model,
        'best_checkpoint': best_path,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'confusion_matrix': cm,
    }
