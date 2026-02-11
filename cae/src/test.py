"""
CAE Test Script - Evaluate Trained Model on Test Set

Usage:
    python test.py --model_path ./models/best_cae_model.pth --dataset_dir ./dataset
    python test.py --model_path ./models/best_cae_model.pth --dataset_dir ./dataset --visualize
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix, classification_report
)

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import CAE, CAELarge, CAESmall, CAETiny
from dataset import AnomalyDataset, get_transforms, denormalize


def load_model(model_path, device):
    """Load trained CAE model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    latent_dim = config.get('latent_dim', 128)
    model_type = config.get('model_type', 'standard')
    
    # Create model based on type
    if model_type == 'large':
        model = CAELarge(latent_dim=latent_dim)
    elif model_type == 'small':
        model = CAESmall(latent_dim=min(latent_dim, 64))
    elif model_type == 'tiny':
        model = CAETiny(latent_dim=min(latent_dim, 32))
    else:
        model = CAE(latent_dim=latent_dim)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    threshold = checkpoint.get('threshold', 0.00001)
    
    print(f"Model loaded from: {model_path}")
    print(f"  Model type: {model_type}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Threshold: {threshold:.6f}")
    
    return model, threshold, config


def evaluate(model, test_loader, threshold, device):
    """Evaluate model on test set."""
    model.eval()
    
    all_errors = []
    all_labels = []
    all_defect_types = []
    all_paths = []
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for images, labels, defect_types, paths in tqdm(test_loader):
            images = images.to(device)
            
            # Get reconstruction error
            reconstructions = model(images)
            errors = torch.mean((images - reconstructions) ** 2, dim=[1, 2, 3])
            
            all_errors.extend(errors.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_defect_types.extend(defect_types)
            all_paths.extend(paths)
    
    all_errors = np.array(all_errors)
    all_labels = np.array(all_labels)
    all_defect_types = np.array(all_defect_types)
    
    # Compute predictions
    predictions = (all_errors > threshold).astype(int)
    
    # Metrics
    results = {
        'auc': roc_auc_score(all_labels, all_errors),
        'f1': f1_score(all_labels, predictions),
        'accuracy': accuracy_score(all_labels, predictions),
        'precision': precision_score(all_labels, predictions),
        'recall': recall_score(all_labels, predictions),
        'threshold': threshold,
        'errors': all_errors,
        'labels': all_labels,
        'predictions': predictions,
        'defect_types': all_defect_types,
        'paths': all_paths
    }
    
    return results


def print_results(results):
    """Print evaluation results."""
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"AUC-ROC Score:  {results['auc']:.4f}")
    print(f"F1 Score:       {results['f1']:.4f}")
    print(f"Accuracy:       {results['accuracy']:.4f}")
    print(f"Precision:      {results['precision']:.4f}")
    print(f"Recall:         {results['recall']:.4f}")
    print(f"Threshold:      {results['threshold']:.6f}")
    
    # Confusion Matrix
    cm = confusion_matrix(results['labels'], results['predictions'])
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              Normal  Anomaly")
    print(f"  Actual Normal  {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"  Actual Anomaly {cm[1,0]:5d}   {cm[1,1]:5d}")
    
    # Per-class detection rate
    print(f"\nPer-Class Detection Rate:")
    for dtype in ['ND', 'CR', 'LP', 'PO']:
        mask = results['defect_types'] == dtype
        if mask.sum() > 0:
            if dtype == 'ND':
                # Normal should be predicted as 0 (not anomaly)
                rate = np.mean(results['predictions'][mask] == 0)
                print(f"  {dtype} (Normal):  {rate:.2%} correctly identified as normal")
            else:
                # Defects should be predicted as 1 (anomaly)
                rate = np.mean(results['predictions'][mask] == 1)
                print(f"  {dtype} (Defect):  {rate:.2%} detected as anomaly")
    
    # Error statistics
    normal_errors = results['errors'][results['labels'] == 0]
    defect_errors = results['errors'][results['labels'] == 1]
    
    print(f"\nReconstruction Error Statistics:")
    print(f"  Normal:  mean={normal_errors.mean():.6f}, std={normal_errors.std():.6f}")
    print(f"  Defect:  mean={defect_errors.mean():.6f}, std={defect_errors.std():.6f}")
    print(f"  Separation: {defect_errors.mean() - normal_errors.mean():.6f}")


def plot_results(results, output_dir):
    """Generate and save visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. ROC Curve
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    fpr, tpr, thresholds = roc_curve(results['labels'], results['errors'])
    axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f"AUC = {results['auc']:.4f}")
    axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=1)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Error Distribution
    normal_errors = results['errors'][results['labels'] == 0]
    defect_errors = results['errors'][results['labels'] == 1]
    
    axes[0, 1].hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='green')
    axes[0, 1].hist(defect_errors, bins=50, alpha=0.7, label='Defect', color='red')
    axes[0, 1].axvline(results['threshold'], color='black', linestyle='--', 
                       linewidth=2, label=f"Threshold={results['threshold']:.5f}")
    axes[0, 1].set_xlabel('Reconstruction Error (MSE)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Per-class error boxplot
    defect_types = ['ND', 'CR', 'LP', 'PO']
    errors_by_type = []
    labels_by_type = []
    for dtype in defect_types:
        mask = results['defect_types'] == dtype
        if mask.sum() > 0:
            errors_by_type.append(results['errors'][mask])
            labels_by_type.append(dtype)
    
    bp = axes[1, 0].boxplot(errors_by_type, labels=labels_by_type, patch_artist=True)
    colors = ['green', 'red', 'orange', 'purple']
    for patch, color in zip(bp['boxes'], colors[:len(labels_by_type)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 0].axhline(results['threshold'], color='black', linestyle='--', 
                       linewidth=2, label='Threshold')
    axes[1, 0].set_xlabel('Defect Type')
    axes[1, 0].set_ylabel('Reconstruction Error')
    axes[1, 0].set_title('Error by Defect Type')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Detection rate bar chart
    detection_rates = []
    for dtype in defect_types:
        mask = results['defect_types'] == dtype
        if mask.sum() > 0:
            if dtype == 'ND':
                rate = np.mean(results['predictions'][mask] == 0)
            else:
                rate = np.mean(results['predictions'][mask] == 1)
            detection_rates.append(rate * 100)
        else:
            detection_rates.append(0)
    
    bars = axes[1, 1].bar(defect_types, detection_rates, color=colors)
    axes[1, 1].axhline(80, color='red', linestyle='--', alpha=0.5, label='80% target')
    axes[1, 1].set_xlabel('Defect Type')
    axes[1, 1].set_ylabel('Detection Rate (%)')
    axes[1, 1].set_title('Detection Rate by Type')
    axes[1, 1].set_ylim(0, 105)
    for bar, rate in zip(bars, detection_rates):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate:.1f}%', ha='center', fontsize=10)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_results.png', dpi=150)
    plt.show()
    print(f"\nPlots saved to: {output_dir / 'test_results.png'}")


def visualize_samples(model, test_loader, results, output_dir, device, n_samples=8):
    """Visualize sample reconstructions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Get some samples
    images, labels, defect_types, paths = next(iter(test_loader))
    images = images[:n_samples].to(device)
    
    with torch.no_grad():
        reconstructions = model(images)
        
        # Compute error on normalized images (this is what the model sees)
        errors = torch.mean((images - reconstructions) ** 2, dim=1)  # Per-pixel error
    
    # Denormalize for visualization
    images_vis = denormalize(images.cpu())
    recon_vis = denormalize(reconstructions.cpu())
    recon_vis = torch.clamp(recon_vis, 0, 1)  # Ensure valid range
    
    # Plot
    fig, axes = plt.subplots(3, n_samples, figsize=(2*n_samples, 6))
    
    for i in range(n_samples):
        # Original
        img_np = images_vis[i].permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f'{defect_types[i]}', fontsize=10)
        axes[0, i].axis('off')
        
        # Reconstruction
        recon_np = recon_vis[i].permute(1, 2, 0).numpy()
        recon_np = np.clip(recon_np, 0, 1)
        axes[1, i].imshow(recon_np)
        axes[1, i].axis('off')
        
        # Error heatmap (normalized per-image for better visualization)
        error_map = errors[i].cpu().numpy()
        # Normalize to [0, 1] for this specific image
        error_min = error_map.min()
        error_max = error_map.max()
        if error_max > error_min:
            error_map = (error_map - error_min) / (error_max - error_min)
        axes[2, i].imshow(error_map, cmap='hot', vmin=0, vmax=1)
        axes[2, i].axis('off')
    
    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('Reconstruction', fontsize=12)
    axes[2, 0].set_ylabel('Error Map', fontsize=12)
    
    plt.suptitle('CAE Reconstruction Samples', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'reconstruction_samples.png', dpi=150)
    plt.show()
    print(f"Samples saved to: {output_dir / 'reconstruction_samples.png'}")


def find_best_threshold(results):
    """Find optimal threshold that maximizes F1 score."""
    errors = results['errors']
    labels = results['labels']
    
    best_f1 = 0
    best_threshold = results['threshold']
    
    for threshold in np.linspace(errors.min(), errors.max(), 200):
        predictions = (errors > threshold).astype(int)
        f1 = f1_score(labels, predictions)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nOptimal Threshold Search:")
    print(f"  Current threshold: {results['threshold']:.6f} (F1={results['f1']:.4f})")
    print(f"  Optimal threshold: {best_threshold:.6f} (F1={best_f1:.4f})")
    
    return best_threshold, best_f1


def save_all_visualizations(model, test_loader, threshold, output_dir, device):
    """
    Save original, reconstructed, and anomaly heatmap side-by-side for ALL images.
    Organized by subclass folders: results/CR/, results/LP/, results/ND/, results/PO/
    """
    output_dir = Path(output_dir)
    
    # Create subclass folders
    for subclass in ['CR', 'LP', 'ND', 'PO']:
        (output_dir / subclass).mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    print(f"\nSaving all visualizations to {output_dir}/[CR,LP,ND,PO]/...")
    
    with torch.no_grad():
        for images, labels, defect_types, paths in tqdm(test_loader, desc="Saving images"):
            images = images.to(device)
            reconstructions = model(images)
            
            # Compute per-pixel error
            errors = torch.mean((images - reconstructions) ** 2, dim=1)  # (B, H, W)
            
            # Denormalize for visualization
            images_vis = denormalize(images.cpu())
            recon_vis = denormalize(reconstructions.cpu())
            recon_vis = torch.clamp(recon_vis, 0, 1)
            
            # Compute scalar error for each image
            scalar_errors = torch.mean(errors, dim=[1, 2]).cpu().numpy()
            
            for i in range(len(images)):
                defect_type = defect_types[i]
                img_path = Path(paths[i])
                img_name = img_path.stem
                error_value = scalar_errors[i]
                is_anomaly = error_value > threshold
                
                # Create figure with 3 subplots side by side
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                
                # Original
                img_np = images_vis[i].permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 1)
                axes[0].imshow(img_np)
                axes[0].set_title(f'Original ({defect_type})', fontsize=12)
                axes[0].axis('off')
                
                # Reconstruction
                recon_np = recon_vis[i].permute(1, 2, 0).numpy()
                recon_np = np.clip(recon_np, 0, 1)
                axes[1].imshow(recon_np)
                axes[1].set_title('Reconstructed', fontsize=12)
                axes[1].axis('off')
                
                # Error heatmap with enhanced visualization
                error_map = errors[i].cpu().numpy()
                error_min = error_map.min()
                error_max = error_map.max()
                if error_max > error_min:
                    error_map_norm = (error_map - error_min) / (error_max - error_min)
                else:
                    error_map_norm = np.zeros_like(error_map)
                
                # Apply gamma correction to enhance visibility (gamma < 1 brightens)
                gamma = 0.5
                error_map_enhanced = np.power(error_map_norm, gamma)
                
                # Create overlay: original image with heatmap on top
                axes[2].imshow(img_np)
                im = axes[2].imshow(error_map_enhanced, cmap='jet', alpha=0.6, vmin=0, vmax=1)
                status = "ANOMALY" if is_anomaly else "NORMAL"
                color = "red" if is_anomaly else "green"
                axes[2].set_title(f'Anomaly Overlay\nError: {error_value:.6f} [{status}]', 
                                  fontsize=11, color=color)
                axes[2].axis('off')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                
                # Save to appropriate subclass folder
                save_path = output_dir / defect_type / f'{img_name}.png'
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
    
    # Count saved files
    for subclass in ['CR', 'LP', 'ND', 'PO']:
        count = len(list((output_dir / subclass).glob('*.png')))
        print(f"  {subclass}/: {count} images saved")
    
    print(f"\nAll visualizations saved to: {output_dir}")


def save_samples_per_class(model, test_loader, threshold, output_dir, device, num_samples=10):
    """
    Save N samples from each class (CR, LP, ND, PO) with original/reconstructed/heatmap.
    """
    output_dir = Path(output_dir)
    
    # Create subclass folders
    for subclass in ['CR', 'LP', 'ND', 'PO']:
        (output_dir / subclass).mkdir(parents=True, exist_ok=True)
    
    # Track how many saved per class
    saved_counts = {'CR': 0, 'LP': 0, 'ND': 0, 'PO': 0}
    
    model.eval()
    
    print(f"\nSaving {num_samples} samples per class to {output_dir}/...")
    
    with torch.no_grad():
        for images, labels, defect_types, paths in tqdm(test_loader, desc="Processing"):
            # Check if we have enough for all classes
            if all(c >= num_samples for c in saved_counts.values()):
                break
                
            images = images.to(device)
            reconstructions = model(images)
            
            # Compute per-pixel error
            errors = torch.mean((images - reconstructions) ** 2, dim=1)  # (B, H, W)
            
            # Denormalize for visualization
            images_vis = denormalize(images.cpu())
            recon_vis = denormalize(reconstructions.cpu())
            recon_vis = torch.clamp(recon_vis, 0, 1)
            
            # Compute scalar error for each image
            scalar_errors = torch.mean(errors, dim=[1, 2]).cpu().numpy()
            
            for i in range(len(images)):
                defect_type = defect_types[i]
                
                # Skip if we already have enough of this class
                if saved_counts[defect_type] >= num_samples:
                    continue
                
                img_path = Path(paths[i])
                img_name = img_path.stem
                error_value = scalar_errors[i]
                is_anomaly = error_value > threshold
                
                # Create figure with 3 subplots side by side
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                
                # Original
                img_np = images_vis[i].permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 1)
                axes[0].imshow(img_np)
                axes[0].set_title(f'Original ({defect_type})', fontsize=12)
                axes[0].axis('off')
                
                # Reconstruction
                recon_np = recon_vis[i].permute(1, 2, 0).numpy()
                recon_np = np.clip(recon_np, 0, 1)
                axes[1].imshow(recon_np)
                axes[1].set_title('Reconstructed', fontsize=12)
                axes[1].axis('off')
                
                # Error heatmap with enhanced visualization
                error_map = errors[i].cpu().numpy()
                error_min = error_map.min()
                error_max = error_map.max()
                if error_max > error_min:
                    error_map_norm = (error_map - error_min) / (error_max - error_min)
                else:
                    error_map_norm = np.zeros_like(error_map)
                
                # Apply gamma correction to enhance visibility (gamma < 1 brightens)
                gamma = 0.5
                error_map_enhanced = np.power(error_map_norm, gamma)
                
                # Create overlay: original image with heatmap on top
                axes[2].imshow(img_np)
                im = axes[2].imshow(error_map_enhanced, cmap='jet', alpha=0.6, vmin=0, vmax=1)
                status = "ANOMALY" if is_anomaly else "NORMAL"
                color = "red" if is_anomaly else "green"
                axes[2].set_title(f'Anomaly Overlay\nError: {error_value:.6f} [{status}]', 
                                  fontsize=11, color=color)
                axes[2].axis('off')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                
                # Save to appropriate subclass folder
                save_path = output_dir / defect_type / f'{img_name}.png'
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                saved_counts[defect_type] += 1
    
    # Report saved counts
    print(f"\nSaved samples per class:")
    for subclass in ['CR', 'LP', 'ND', 'PO']:
        print(f"  {subclass}/: {saved_counts[subclass]} images")
    
    print(f"\nVisualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Test CAE Model')
    parser.add_argument('--model_path', type=str, default='./models/best_cae_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--dataset_dir', type=str, default='./dataset',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--save_all', action='store_true',
                        help='Save original/reconstructed/heatmap for ALL images by subclass')
    parser.add_argument('--save_samples', type=int, default=0,
                        help='Save N samples per class (e.g., --save_samples 10)')
    parser.add_argument('--find_threshold', action='store_true',
                        help='Search for optimal threshold')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override threshold (use optimal from --find_threshold)')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, threshold, config = load_model(args.model_path, device)
    
    # Override threshold if specified
    if args.threshold is not None:
        print(f"Using override threshold: {args.threshold:.6f}")
        threshold = args.threshold
    
    # Use image size from config if available
    image_size = config.get('image_size', args.image_size)
    
    # Create test dataset
    test_transform = get_transforms(image_size, augment=False)
    test_dataset = AnomalyDataset(
        Path(args.dataset_dir) / 'testing',
        transform=test_transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == 'cuda'
    )
    
    # Evaluate
    results = evaluate(model, test_loader, threshold, device)
    
    # Print results
    print_results(results)
    
    # Find optimal threshold
    if args.find_threshold:
        find_best_threshold(results)
    
    # Visualizations
    if args.visualize:
        plot_results(results, args.output_dir)
        visualize_samples(model, test_loader, results, args.output_dir, device)
    
    # Save all visualizations by subclass
    if args.save_all:
        save_all_visualizations(model, test_loader, threshold, args.output_dir, device)
    
    # Save N samples per class
    if args.save_samples > 0:
        save_samples_per_class(model, test_loader, threshold, args.output_dir, device, args.save_samples)
    
    print("\nTest complete!")


if __name__ == '__main__':
    main()
