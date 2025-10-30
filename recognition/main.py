import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# Custom ConvNeXt-inspired Block Implementation
class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for images"""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block with depthwise conv and inverted bottleneck"""

    def __init__(self, dim, drop_path=0., layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = nn.Parameter(layer_scale_init * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init > 0 else None
        self.drop_path = nn.Dropout2d(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        x = input + self.drop_path(x)
        return x


class CustomConvNeXt(nn.Module):
    """Custom ConvNeXt-inspired architecture for AD classification"""

    def __init__(self, in_chans=3, num_classes=2, depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768], drop_path_rate=0.2):
        super().__init__()
        self.downsample_layers = nn.ModuleList()

        # Stem
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)

        # Downsample layers between stages
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Stage blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j])
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Head
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = x.mean([-2, -1])  # Global average pooling
        x = self.norm(x)
        x = self.head(x)
        return x


class ADNIDataset(Dataset):
    """Custom dataset for ADNI brain MRI images"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            # Create placeholder if image not found
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


def prepare_data(data_dir, test_size=0.2, val_size=0.1):
    """Prepare data splits from actual directory structure:
    data_dir/
        ├── train/
        │   ├── NC/
        │   └── AD/
        └── test/
            ├── NC/
            └── AD/
    """

    image_paths_train = []
    labels_train = []
    image_paths_test = []
    labels_test = []

    class_map = {'NC': 0, 'AD': 1}

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    for class_name, class_idx in class_map.items():
        class_train_path = os.path.join(train_dir, class_name)
        if os.path.exists(class_train_path):
            for img_name in os.listdir(class_train_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg', '.nii', '.nii.gz')):
                    image_paths_train.append(os.path.join(class_train_path, img_name))
                    labels_train.append(class_idx)

    for class_name, class_idx in class_map.items():
        class_test_path = os.path.join(test_dir, class_name)
        if os.path.exists(class_test_path):
            for img_name in os.listdir(class_test_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg', '.nii', '.nii.gz')):
                    image_paths_test.append(os.path.join(class_test_path, img_name))
                    labels_test.append(class_idx)

    if len(image_paths_train) == 0 and len(image_paths_test) == 0:
        print("No data found. Creating synthetic dataset for demonstration...")
        n_samples = 1000
        n_test = int(n_samples * test_size)
        n_train_temp = n_samples - n_test

        image_paths_train = [f"synthetic_train_{i}.jpg" for i in range(n_train_temp)]
        labels_train = np.random.randint(0, 2, n_train_temp).tolist()

        image_paths_test = [f"synthetic_test_{i}.jpg" for i in range(n_test)]
        labels_test = np.random.randint(0, 2, n_test).tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        image_paths_train, labels_train,
        test_size=val_size / (1 - test_size),
        random_state=42,
        stratify=labels_train
    )

    X_test, y_test = image_paths_test, labels_test

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
    """Create data loaders with augmentation"""

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # No augmentation for validation/test
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = ADNIDataset(X_train, y_train, transform=train_transform)
    val_dataset = ADNIDataset(X_val, y_val, transform=test_transform)
    test_dataset = ADNIDataset(X_test, y_test, transform=test_transform)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(targets, predictions)

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(targets, predictions)

    return epoch_loss, epoch_acc, predictions, targets


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    DATA_DIR = "path/to/adni/data"  # Update this path

    # Prepare data
    print("Preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(DATA_DIR)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, BATCH_SIZE
    )

    # Initialize model
    print("Initializing model...")
    model = CustomConvNeXt(
        in_chans=3,
        num_classes=2,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768]
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    # Training loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with Val Acc: {val_acc:.4f}")

    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)

    # Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc, test_preds, test_targets = evaluate(model, test_loader, criterion, device)

    print(f"\n{'=' * 50}")
    print(f"FINAL TEST ACCURACY: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"{'=' * 50}")

    # Confusion matrix and classification report
    cm = confusion_matrix(test_targets, test_preds)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds,
                                target_names=['Normal', 'AD']))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal', 'AD'])
    plt.yticks(tick_marks, ['Normal', 'AD'])
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    return model, test_acc


if __name__ == "__main__":
    model, test_accuracy = main()

    if test_accuracy >= 0.8:
        print(f"\n✓ Target accuracy of 0.8 achieved! Test accuracy: {test_accuracy:.4f}")
    else:
        print(f"\n✗ Target accuracy not met. Test accuracy: {test_accuracy:.4f}")
        print("Consider: increasing epochs, adjusting learning rate, or adding more data augmentation")