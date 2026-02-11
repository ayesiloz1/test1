"""
CAE Training Script for Weld Defect Detection

Training Strategy:
1. Train on normal images only (unsupervised)
2. Validate by computing reconstruction error on normal + defects
3. Find optimal threshold that separates normal from defects
4. Test final performance with the learned threshold
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import CAE, CAELarge, CAESmall, CAETiny
from dataset import create_dataloaders, denormalize

# Import security utils for secure model saving
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'gui'))
try:
    from security_utils import ModelIntegrityVerifier
    SECURITY_UTILS_AVAILABLE = True
except ImportError:
    SECURITY_UTILS_AVAILABLE = False
    print("Note: Security utils not available. Model integrity verification disabled.")


class SSIMLoss(nn.Module):
    """
    SSIM (Structural Similarity Index) Loss for better defect localization.
    SSIM captures structural differences (edges, textures) better than MSE.
    """
    def __init__(self, window_size=11, channels=3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        
        # Create Gaussian window
        sigma = 1.5
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2 * sigma**2)) 
                              for x in range(window_size)])
        gauss = gauss / gauss.sum()
        
        # Create 2D kernel
        kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        kernel = kernel.expand(channels, 1, window_size, window_size).contiguous()
        self.register_buffer('window', kernel)
        
    def forward(self, img1, img2):
        """Compute SSIM loss (1 - SSIM) so lower is better."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Compute means
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channels)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channels)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=self.channels) - mu1_mu2
        
        # SSIM formula
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Return 1 - SSIM as loss (so minimizing loss maximizes SSIM)
        return 1 - ssim.mean()


class CombinedLoss(nn.Module):
    """Combined MSE + SSIM loss for both pixel accuracy and structural similarity."""
    def __init__(self, alpha=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()
        self.alpha = alpha  # Weight for SSIM (0.5 = equal weight)
        
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        ssim_loss = self.ssim(pred, target)
        return (1 - self.alpha) * mse_loss + self.alpha * ssim_loss


class CAETrainer:
    """Trainer class for Convolutional Autoencoder"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create model
        if config.model_type == 'large':
            self.model = CAELarge(latent_dim=config.latent_dim).to(self.device)
        elif config.model_type == 'small':
            self.model = CAESmall(latent_dim=min(config.latent_dim, 64)).to(self.device)
        elif config.model_type == 'tiny':
            self.model = CAETiny(latent_dim=min(config.latent_dim, 32)).to(self.device)
        else:
            self.model = CAE(latent_dim=config.latent_dim).to(self.device)
        
        print(f"Model: {config.model_type.upper()}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Loss function
        if config.loss_type == 'ssim':
            self.criterion = SSIMLoss().to(self.device)
            print(f"Loss: SSIM (structural similarity)")
        elif config.loss_type == 'combined':
            self.criterion = CombinedLoss(alpha=config.ssim_weight).to(self.device)
            print(f"Loss: Combined (MSE + SSIM, alpha={config.ssim_weight})")
        else:
            self.criterion = nn.MSELoss()
            print(f"Loss: MSE")
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Create dataloaders
        # Use pin_memory=False for CPU training
        use_pin_memory = self.device.type == 'cuda'
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            config.dataset_dir,
            batch_size=config.batch_size,
            image_size=config.image_size,
            num_workers=config.num_workers,
            pin_memory=use_pin_memory
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'threshold': []
        }
        
        # Best model tracking
        self.best_val_auc = 0
        self.best_threshold = 0
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, _ in pbar:
            images = images.to(self.device)
            
            # Forward pass
            reconstructions = self.model(images)
            loss = self.criterion(reconstructions, images)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate and compute AUC score"""
        self.model.eval()
        
        all_errors = []
        all_labels = []
        all_defect_types = []
        
        with torch.no_grad():
            for images, labels, defect_types, _ in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                
                # Get reconstruction error
                reconstructions = self.model(images)
                errors = torch.mean((images - reconstructions) ** 2, dim=[1, 2, 3])
                
                all_errors.extend(errors.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_defect_types.extend(defect_types)
        
        all_errors = np.array(all_errors)
        all_labels = np.array(all_labels)
        
        # Compute validation loss (average error on normal samples)
        normal_mask = all_labels == 0
        val_loss = np.mean(all_errors[normal_mask])
        
        # Compute AUC score
        if len(np.unique(all_labels)) > 1:
            auc_score = roc_auc_score(all_labels, all_errors)
        else:
            auc_score = 0.5
        
        # Find optimal threshold using F1 score
        thresholds = np.percentile(all_errors[normal_mask], [90, 95, 99])
        best_f1 = 0
        best_threshold = thresholds[1]  # Default to 95th percentile
        
        for thresh in np.linspace(all_errors.min(), all_errors.max(), 100):
            predictions = (all_errors > thresh).astype(int)
            f1 = f1_score(all_labels, predictions, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        return val_loss, auc_score, best_threshold, all_errors, all_labels, all_defect_types
    
    def train(self):
        """Full training loop"""
        print(f"\nStarting training for {self.config.epochs} epochs...")
        print("=" * 60)
        
        for epoch in range(1, self.config.epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print("-" * 40)
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_auc, threshold, errors, labels, defect_types = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            self.history['threshold'].append(threshold)
            
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss:   {val_loss:.6f}")
            print(f"Val AUC:    {val_auc:.4f}")
            print(f"Threshold:  {threshold:.6f}")
            
            # Save best model
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_threshold = threshold
                self.save_model('best_model.pth')
                print(f"  -> New best model! AUC: {val_auc:.4f}")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch}.pth')
        
        # Save final model
        self.save_model('final_model.pth')
        
        # Plot training history
        self.plot_history()
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best Val AUC: {self.best_val_auc:.4f}")
        print(f"Best Threshold: {self.best_threshold:.6f}")
        
        return self.best_val_auc, self.best_threshold
    
    def test(self, threshold=None):
        """Test the model on test set"""
        if threshold is None:
            threshold = self.best_threshold
        
        print(f"\nTesting with threshold: {threshold:.6f}")
        print("-" * 40)
        
        self.model.eval()
        
        all_errors = []
        all_labels = []
        all_defect_types = []
        
        with torch.no_grad():
            for images, labels, defect_types, _ in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device)
                
                reconstructions = self.model(images)
                errors = torch.mean((images - reconstructions) ** 2, dim=[1, 2, 3])
                
                all_errors.extend(errors.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_defect_types.extend(defect_types)
        
        all_errors = np.array(all_errors)
        all_labels = np.array(all_labels)
        all_defect_types = np.array(all_defect_types)
        
        # Predictions
        predictions = (all_errors > threshold).astype(int)
        
        # Metrics
        auc_score = roc_auc_score(all_labels, all_errors)
        f1 = f1_score(all_labels, predictions)
        
        # Confusion matrix values
        tp = np.sum((predictions == 1) & (all_labels == 1))
        tn = np.sum((predictions == 0) & (all_labels == 0))
        fp = np.sum((predictions == 1) & (all_labels == 0))
        fn = np.sum((predictions == 0) & (all_labels == 1))
        
        accuracy = (tp + tn) / len(all_labels)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\nTest Results:")
        print(f"  AUC Score:  {auc_score:.4f}")
        print(f"  F1 Score:   {f1:.4f}")
        print(f"  Accuracy:   {accuracy:.4f}")
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        
        # Per-defect-type analysis
        print(f"\nPer-Defect-Type Analysis:")
        for defect_type in ['ND', 'CR', 'LP', 'PO']:
            mask = all_defect_types == defect_type
            if np.sum(mask) > 0:
                type_errors = all_errors[mask]
                type_preds = predictions[mask]
                if defect_type == 'ND':
                    correct = np.sum(type_preds == 0)
                else:
                    correct = np.sum(type_preds == 1)
                print(f"  {defect_type}: {correct}/{np.sum(mask)} correct ({100*correct/np.sum(mask):.1f}%)")
                print(f"       Mean error: {np.mean(type_errors):.6f}, Std: {np.std(type_errors):.6f}")
        
        # Plot ROC curve
        self.plot_roc_curve(all_labels, all_errors)
        
        # Plot error distribution
        self.plot_error_distribution(all_errors, all_labels, all_defect_types, threshold)
        
        return {
            'auc': auc_score,
            'f1': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'threshold': threshold
        }
    
    def save_model(self, filename):
        """Save model checkpoint with automatic integrity hash generation"""
        save_path = self.output_dir / filename
        
        checkpoint_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': vars(self.config),
            'best_val_auc': self.best_val_auc,
            'best_threshold': self.best_threshold,
            'history': self.history
        }
        
        # Save securely with automatic hash generation
        if SECURITY_UTILS_AVAILABLE:
            verifier = ModelIntegrityVerifier()
            verifier.save_pytorch_model(checkpoint_data, str(save_path), is_state_dict=False)
            print(f"✓ Model saved securely: {save_path}")
            print(f"✓ Integrity hash: {save_path}.hash")
        else:
            torch.save(checkpoint_data, save_path)
            print(f"Model saved to {save_path}")
            print(f"⚠ Warning: No integrity hash generated (security_utils not available)")
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_auc = checkpoint.get('best_val_auc', 0)
        self.best_threshold = checkpoint.get('best_threshold', 0)
        self.history = checkpoint.get('history', self.history)
        print(f"Model loaded from {filepath}")
        print(f"  Best AUC: {self.best_val_auc:.4f}, Threshold: {self.best_threshold:.6f}")
    
    def plot_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # AUC
        axes[1].plot(self.history['val_auc'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('Validation AUC')
        axes[1].grid(True)
        
        # Threshold
        axes[2].plot(self.history['threshold'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Threshold')
        axes[2].set_title('Optimal Threshold')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150)
        plt.close()
        print(f"Training history saved to {self.output_dir / 'training_history.png'}")
    
    def plot_roc_curve(self, labels, scores):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Anomaly Detection')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(self.output_dir / 'roc_curve.png', dpi=150)
        plt.close()
        print(f"ROC curve saved to {self.output_dir / 'roc_curve.png'}")
    
    def plot_error_distribution(self, errors, labels, defect_types, threshold):
        """Plot reconstruction error distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # By normal vs defect
        normal_errors = errors[labels == 0]
        defect_errors = errors[labels == 1]
        
        axes[0].hist(normal_errors, bins=50, alpha=0.7, label=f'Normal (n={len(normal_errors)})', color='green')
        axes[0].hist(defect_errors, bins=50, alpha=0.7, label=f'Defect (n={len(defect_errors)})', color='red')
        axes[0].axvline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.4f}')
        axes[0].set_xlabel('Reconstruction Error')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Error Distribution: Normal vs Defect')
        axes[0].legend()
        axes[0].grid(True)
        
        # By defect type
        colors = {'ND': 'green', 'CR': 'red', 'LP': 'orange', 'PO': 'purple'}
        for defect_type in ['ND', 'CR', 'LP', 'PO']:
            mask = defect_types == defect_type
            if np.sum(mask) > 0:
                axes[1].hist(errors[mask], bins=30, alpha=0.6, 
                           label=f'{defect_type} (n={np.sum(mask)})', color=colors[defect_type])
        axes[1].axvline(threshold, color='black', linestyle='--', label=f'Threshold')
        axes[1].set_xlabel('Reconstruction Error')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Error Distribution by Defect Type')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_distribution.png', dpi=150)
        plt.close()
        print(f"Error distribution saved to {self.output_dir / 'error_distribution.png'}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train CAE for Weld Defect Detection')
    
    # Dataset
    parser.add_argument('--dataset_dir', type=str, default='../dataset',
                        help='Path to CAE dataset')
    
    # Model
    parser.add_argument('--model_type', type=str, default='standard', 
                        choices=['standard', 'large', 'small', 'tiny'],
                        help='Model type: standard (~886K), large (~2.7M), small (~110K), tiny (~30K)')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent dimension size')
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'ssim', 'combined'],
                        help='Loss function: mse, ssim, or combined')
    parser.add_argument('--ssim_weight', type=float, default=0.5,
                        help='Weight for SSIM in combined loss (0-1)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    
    # Data
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='../models',
                        help='Output directory for models')
    
    # Mode
    parser.add_argument('--test_only', action='store_true',
                        help='Only run testing')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model for testing')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Convert relative paths to absolute
    script_dir = Path(__file__).parent
    if not Path(args.dataset_dir).is_absolute():
        args.dataset_dir = str(script_dir / args.dataset_dir)
    if not Path(args.output_dir).is_absolute():
        args.output_dir = str(script_dir / args.output_dir)
    
    print("=" * 60)
    print("CAE Training for Weld Defect Detection")
    print("=" * 60)
    print(f"\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # Create trainer
    trainer = CAETrainer(args)
    
    if args.test_only:
        # Load model and test
        if args.model_path:
            trainer.load_model(args.model_path)
        else:
            trainer.load_model(Path(args.output_dir) / 'best_model.pth')
        trainer.test()
    else:
        # Train and test
        trainer.train()
        trainer.test()


if __name__ == "__main__":
    main()
