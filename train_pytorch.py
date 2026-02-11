"""
Welding Defect Classification using PyTorch - Training from Scratch
Based on RIAWELC Dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from tqdm import tqdm
import time
import sys
from pathlib import Path

# Add gui directory to path for security utils
sys.path.append(str(Path(__file__).parent / 'gui'))
try:
    from security_utils import ModelIntegrityVerifier
    SECURITY_UTILS_AVAILABLE = True
except ImportError:
    SECURITY_UTILS_AVAILABLE = False
    print("Note: Security utils not available. Model integrity verification disabled.")

# Add gui directory to path for security utils
sys.path.append(str(Path(__file__).parent / 'gui'))
try:
    from security_utils import ModelIntegrityVerifier
    SECURITY_UTILS_AVAILABLE = True
except ImportError:
    SECURITY_UTILS_AVAILABLE = False
    print("Note: Security utils not available. Model integrity verification disabled.")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Dataset paths
    DATASET_PATH = 'dataset'
    TRAIN_DIR = os.path.join(DATASET_PATH, 'training')
    VAL_DIR = os.path.join(DATASET_PATH, 'validation')
    TEST_DIR = os.path.join(DATASET_PATH, 'testing')
    
    # Training parameters
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4  # Adjust based on your CPU
    
    # Classes
    CLASSES = ['CR', 'LP', 'ND', 'PO']
    NUM_CLASSES = 4
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model save path
    MODEL_SAVE_PATH = 'models/weld_defect_pytorch.pth'
    CHECKPOINT_PATH = 'models/best_model_pytorch.pth'

# ============================================================================
# DATA PREPARATION
# ============================================================================

def get_data_transforms():
    """Define data augmentation and normalization"""
    
    # Training transforms with heavy augmentation
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.3, 0.3)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/Test transforms (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

def create_data_loaders():
    """Create PyTorch data loaders"""
    
    train_transform, val_test_transform = get_data_transforms()
    
    # Load datasets
    train_dataset = datasets.ImageFolder(Config.TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(Config.VAL_DIR, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(Config.TEST_DIR, transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN - Good starting point"""
    
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class DeepCNN(nn.Module):
    """Deep CNN - Better accuracy"""
    
    def __init__(self, num_classes=4):
        super(DeepCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            # Block 5
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(0.4),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ResidualBlock(nn.Module):
    """Residual block for ResNet-inspired architecture"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResidualCNN(nn.Module):
    """ResNet-inspired CNN with residual connections"""
    
    def __init__(self, num_classes=4):
        super(ResidualCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class LightweightCNN(nn.Module):
    """Lightweight CNN for fast training"""
    
    def __init__(self, num_classes=4):
        super(LightweightCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

class Trainer:
    """Training class with all necessary methods"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, class_names):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=Config.LEARNING_RATE, 
            weight_decay=Config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        self.best_val_acc = 0.0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/len(pbar), 
                            'acc': 100.*correct/total})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validation'):
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def train(self):
        """Full training loop"""
        print(f"\n{'='*70}")
        print(f"Training on {Config.DEVICE}")
        print(f"{'='*70}\n")
        
        for epoch in range(Config.EPOCHS):
            print(f'\nEpoch {epoch+1}/{Config.EPOCHS}')
            print('-' * 70)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f'\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                # Save model securely with automatic hash generation
                if SECURITY_UTILS_AVAILABLE:
                    verifier = ModelIntegrityVerifier()
                    verifier.save_pytorch_model(self.model.state_dict(), Config.CHECKPOINT_PATH)
                else:
                    torch.save(self.model.state_dict(), Config.CHECKPOINT_PATH)
                print(f'✓ Best model saved securely! (Val Acc: {val_acc:.2f}%)')
        
        # Load best model
        self.model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, weights_only=True))
        print(f"\n✓ Training completed! Best validation accuracy: {self.best_val_acc:.2f}%")
    
    def test(self):
        """Test the model"""
        self.model.eval()
        all_preds = []
        all_labels = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc='Testing'):
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        
        print(f"\n{'='*70}")
        print(f"TEST RESULTS")
        print(f"{'='*70}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        # Classification report
        print(f"\n{'='*70}")
        print("CLASSIFICATION REPORT")
        print(f"{'='*70}")
        print(classification_report(all_labels, all_preds, 
                                   target_names=self.class_names, digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        self.plot_confusion_matrix(cm)
        
        return test_acc, all_preds, all_labels
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(self.history['train_acc'], label='Train', linewidth=2)
        axes[0].plot(self.history['val_acc'], label='Validation', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(self.history['train_loss'], label='Train', linewidth=2)
        axes[1].plot(self.history['val_loss'], label='Validation', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history_pytorch.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\n✓ Training history saved as 'training_history_pytorch.png'")
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[0], annot_kws={'size': 12})
        axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        
        # Normalized
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[1], annot_kws={'size': 12})
        axes[1].set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('confusion_matrix_pytorch.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\n✓ Confusion matrix saved as 'confusion_matrix_pytorch.png'")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    print(f"\n{'='*70}")
    print("WELDING DEFECT CLASSIFICATION - PyTorch Implementation")
    print(f"{'='*70}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Load data
    print("\n[STEP 1/5] Loading data...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders()
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    print(f"✓ Classes: {class_names}")
    print(f"✓ Device: {Config.DEVICE}")
    
    # Build model
    print("\n[STEP 2/5] Building model...")
    print("\nAvailable models:")
    print("  1. SimpleCNN - Balanced, good starting point")
    print("  2. DeepCNN - Higher accuracy, longer training")
    print("  3. LightweightCNN - Fastest training")
    print("  4. ResidualCNN - Best for deep architectures")
    
    # Choose model
    model = SimpleCNN(num_classes=Config.NUM_CLASSES)
    # model = DeepCNN(num_classes=Config.NUM_CLASSES)
    # model = LightweightCNN(num_classes=Config.NUM_CLASSES)
    # model = ResidualCNN(num_classes=Config.NUM_CLASSES)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader, class_names)
    
    # Train
    print("\n[STEP 3/5] Training model...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"\n✓ Training time: {training_time/60:.2f} minutes")
    
    # Plot history
    print("\n[STEP 4/5] Plotting training history...")
    trainer.plot_training_history()
    
    # Test
    print("\n[STEP 5/5] Testing model...")
    test_acc, _, _ = trainer.test()
    
    # Save model securely with automatic hash generation
    if SECURITY_UTILS_AVAILABLE:
        verifier = ModelIntegrityVerifier()
        verifier.save_pytorch_model(model.state_dict(), Config.MODEL_SAVE_PATH)
        print(f"\n✓ Model saved securely to {Config.MODEL_SAVE_PATH}")
        print(f"✓ Integrity hash: {Config.MODEL_SAVE_PATH}.hash")
    else:
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        print(f"\n✓ Model saved to {Config.MODEL_SAVE_PATH}")
        print(f"⚠ Warning: Security utils not available - no integrity hash generated")
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE!")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
