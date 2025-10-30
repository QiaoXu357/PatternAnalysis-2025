import os
from typing import Sequence
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from .modules import CustomConvNeXt
from .dataset import PerImageStandardize


def _predict_with_tta(model, images, tta: int = 1):
    """Apply lightweight test-time augmentation (horizontal flip) if enabled."""
    logits = model(images)
    if tta <= 1:
        return logits
    outs = [logits]
    images_flip = torch.flip(images, dims=[3])
    outs.append(model(images_flip))
    return torch.stack(outs, dim=0).mean(0)


def evaluate_checkpoint(data_loader, checkpoint_path: str, model_kwargs: dict | None = None, device: torch.device | None = None, tta: int = 2, label_smoothing: float = 0.1):
    """Load a checkpoint and compute loss/accuracy on a DataLoader."""
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


def predict_image(image_path: str, checkpoint_path: str, model_kwargs: dict | None = None,
                  device: torch.device | None = None, tta: int = 2,
                  class_names: Sequence[str] = ("Normal", "AD")):
    """Predict a single image with a checkpoint and return label and probabilities."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomConvNeXt(**(model_kwargs or {})).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        PerImageStandardize(),
    ])

    with torch.no_grad():
        img = Image.open(image_path).convert('RGB')
        tensor = test_transform(img).unsqueeze(0).to(device)
        logits = _predict_with_tta(model, tensor, tta=tta)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        return {
            'index': pred_idx,
            'label': pred_label,
            'probs': {class_names[i] if i < len(class_names) else str(i): float(probs[i]) for i in range(len(probs))},
        }
