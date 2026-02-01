"""
Dataset and DataLoader for CAE Training

Provides:
- NormalDataset: For training (normal images only)
- AnomalyDataset: For validation/testing (normal + defect with labels)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path


class NormalDataset(Dataset):
    """
    Dataset for training: Contains only normal images
    Used for unsupervised learning where CAE learns normal patterns
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Path to normal images folder (e.g., training/normal)
            transform: Image transformations
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Get all image files
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(self.root_dir.glob(ext))
            self.image_paths.extend(self.root_dir.glob(ext.upper()))
        
        self.image_paths = sorted(self.image_paths)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")
            
        print(f"NormalDataset: Found {len(self.image_paths)} images in {root_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, str(img_path)


class AnomalyDataset(Dataset):
    """
    Dataset for validation/testing: Contains normal and defect images with labels
    
    Labels:
        0: Normal (ND)
        1: Defect (CR, LP, PO)
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Path to validation or testing folder
            transform: Image transformations
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        self.defect_types = []  # Store specific defect type for analysis
        
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        
        # Load normal images (label = 0)
        normal_dir = self.root_dir / 'normal'
        if normal_dir.exists():
            for ext in extensions:
                for img_path in normal_dir.glob(ext):
                    self.image_paths.append(img_path)
                    self.labels.append(0)
                    self.defect_types.append('ND')
                for img_path in normal_dir.glob(ext.upper()):
                    self.image_paths.append(img_path)
                    self.labels.append(0)
                    self.defect_types.append('ND')
        
        # Load defect images (label = 1)
        defect_dir = self.root_dir / 'defect'
        if defect_dir.exists():
            for defect_type in ['CR', 'LP', 'PO']:
                defect_type_dir = defect_dir / defect_type
                if defect_type_dir.exists():
                    for ext in extensions:
                        for img_path in defect_type_dir.glob(ext):
                            self.image_paths.append(img_path)
                            self.labels.append(1)
                            self.defect_types.append(defect_type)
                        for img_path in defect_type_dir.glob(ext.upper()):
                            self.image_paths.append(img_path)
                            self.labels.append(1)
                            self.defect_types.append(defect_type)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        n_normal = sum(1 for l in self.labels if l == 0)
        n_defect = sum(1 for l in self.labels if l == 1)
        print(f"AnomalyDataset: Found {len(self.image_paths)} images ({n_normal} normal, {n_defect} defects) in {root_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        defect_type = self.defect_types[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, defect_type, str(img_path)


# ImageNet normalization constants (same as CNN)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size=224, augment=False):
    """
    Get image transformations (matching CNN preprocessing)
    
    Args:
        image_size: Target image size
        augment: Whether to apply data augmentation (for training)
    
    Returns:
        transforms.Compose object
    """
    if augment:
        # Training transforms with augmentation
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # Validation/Test transforms (no augmentation)
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def denormalize(tensor):
    """
    Denormalize a tensor from ImageNet normalization back to [0, 1] range.
    Used for visualization of CAE reconstructions.
    
    Args:
        tensor: Normalized tensor (C, H, W) or (B, C, H, W)
    
    Returns:
        Denormalized tensor in [0, 1] range
    """
    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    
    if tensor.device.type != 'cpu':
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    
    if tensor.dim() == 4:  # Batch dimension
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    denorm = tensor * std + mean
    return torch.clamp(denorm, 0, 1)


def create_dataloaders(dataset_dir, batch_size=32, image_size=224, num_workers=4, pin_memory=True):
    """
    Create DataLoaders for training, validation, and testing
    
    Args:
        dataset_dir: Path to CAE dataset
        batch_size: Batch size
        image_size: Image size
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for GPU (set False for CPU)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    dataset_dir = Path(dataset_dir)
    
    # Transforms
    train_transform = get_transforms(image_size, augment=True)
    eval_transform = get_transforms(image_size, augment=False)
    
    # Training: Normal images only
    train_dataset = NormalDataset(
        dataset_dir / 'training' / 'normal',
        transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Validation: Normal + Defects
    val_dataset = AnomalyDataset(
        dataset_dir / 'validation',
        transform=eval_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Testing: Normal + Defects
    test_dataset = AnomalyDataset(
        dataset_dir / 'testing',
        transform=eval_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    from pathlib import Path
    
    dataset_dir = Path(__file__).parent.parent / 'dataset'
    
    if not dataset_dir.exists():
        print(f"Dataset not found at {dataset_dir}")
        print("Run create_dataset.py first!")
    else:
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_dir,
            batch_size=4,
            num_workers=0
        )
        
        print(f"\nDataLoader Summary:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Test a batch
        for images, paths in train_loader:
            print(f"\nTrain batch shape: {images.shape}")
            print(f"  Min: {images.min():.3f}, Max: {images.max():.3f}")
            break
        
        for images, labels, defect_types, paths in val_loader:
            print(f"\nVal batch shape: {images.shape}")
            print(f"  Labels: {labels.tolist()}")
            print(f"  Defect types: {defect_types}")
            break
