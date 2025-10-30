import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from .modules import CustomConvNeXt


def _predict_with_tta(model, images, tta: int = 1):
    logits = model(images)
    if tta <= 1:
        return logits
    outs = [logits]
    images_flip = torch.flip(images, dims=[3])
    outs.append(model(images_flip))
    return torch.stack(outs, dim=0).mean(0)


def evaluate_checkpoint(data_loader, checkpoint_path: str, model_kwargs: dict | None = None, device: torch.device | None = None, tta: int = 2, label_smoothing: float = 0.1):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomConvNeXt(**(model_kwargs or {})).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    model.eval()
    running_loss = 0.0
    predictions, targets = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = _predict_with_tta(model, images, tta=tta)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    loss = running_loss / max(1, len(data_loader))
    acc = accuracy_score(targets, predictions) if predictions else 0.0
    cm = confusion_matrix(targets, predictions)
    report = classification_report(targets, predictions, target_names=['Normal', 'AD'])
    return {'loss': loss, 'acc': acc, 'confusion_matrix': cm, 'report': report}
