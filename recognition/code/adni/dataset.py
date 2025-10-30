import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split


class ADNIDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 此处不再用黑图填充，若读取失败交由上游过滤
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


class PerImageStandardize:
    def __call__(self, tensor):
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + 1e-6)


def _is_image_ok(path: str) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def prepare_data(data_dir: str, test_size: float = 0.2, val_size: float = 0.1):
    image_paths_train, labels_train = [], []
    image_paths_test, labels_test = [], []
    class_map = {'NC': 0, 'AD': 1}
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    skipped_train = 0
    for class_name, class_idx in class_map.items():
        class_train_path = os.path.join(train_dir, class_name)
        if os.path.exists(class_train_path):
            for img_name in os.listdir(class_train_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    p = os.path.join(class_train_path, img_name)
                    if _is_image_ok(p):
                        image_paths_train.append(p)
                        labels_train.append(class_idx)
                    else:
                        skipped_train += 1

    skipped_test = 0
    for class_name, class_idx in class_map.items():
        class_test_path = os.path.join(test_dir, class_name)
        if os.path.exists(class_test_path):
            for img_name in os.listdir(class_test_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    p = os.path.join(class_test_path, img_name)
                    if _is_image_ok(p):
                        image_paths_test.append(p)
                        labels_test.append(class_idx)
                    else:
                        skipped_test += 1

    if len(image_paths_train) == 0 and len(image_paths_test) == 0:
        n_samples = 1000
        n_test = int(n_samples * test_size)
        n_train_temp = n_samples - n_test
        image_paths_train = [f"synthetic_train_{i}.jpg" for i in range(n_train_temp)]
        labels_train = np.random.randint(0, 2, n_train_temp).tolist()
        image_paths_test = [f"synthetic_test_{i}.jpg" for i in range(n_test)]
        labels_test = np.random.randint(0, 2, n_test).tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        image_paths_train,
        labels_train,
        test_size=val_size / max(1e-9, (1 - test_size)),
        random_state=42,
        stratify=labels_train if len(set(labels_train)) > 1 else None,
    )
    X_test, y_test = image_paths_test, labels_test
    if skipped_train or skipped_test:
        print(f"Filtered unreadable images -> train:{skipped_train}, test:{skipped_test}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_data_loaders(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    batch_size: int = 32,
    num_workers: int = 2,
    use_weighted_sampler: str = "auto",  # 'auto'|'on'|'off'
):
    # 训练增强：去掉色彩抖动，采用更适合医学图像的增强
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        PerImageStandardize(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    ])

    # 验证/测试：确定性预处理
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        PerImageStandardize(),
    ])

    train_dataset = ADNIDataset(X_train, y_train, transform=train_transform)
    val_dataset = ADNIDataset(X_val, y_val, transform=test_transform)
    test_dataset = ADNIDataset(X_test, y_test, transform=test_transform)

    # 采样策略：按需使用 WeightedRandomSampler
    sampler = None
    if use_weighted_sampler in ("auto", "on"):
        class_counts = {}
        for c in y_train:
            class_counts[c] = class_counts.get(c, 0) + 1
        if class_counts:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / max(1, min_count)
        else:
            imbalance_ratio = 1.0
        need_sampler = use_weighted_sampler == "on" or imbalance_ratio >= 1.5
        if need_sampler:
            weights = [1.0 / class_counts[c] for c in y_train]
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
