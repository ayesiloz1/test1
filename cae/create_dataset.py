"""
CAE Dataset Creator for Weld Defect Detection - FIXED VERSION

This script organizes the dataset for Convolutional Autoencoder training
while PRESERVING the original train/val/test splits to prevent data leakage.

Training Strategy:
- Train CAE on ONLY normal images (learns normal reconstruction)
- Validate with normal + defects (to set threshold)
- Test with normal + defects (to evaluate performance)

IMPORTANT: This version preserves the original CNN dataset splits to prevent leakage!
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Configuration
SEED = 42
random.seed(SEED)

# Paths
BASE_DIR = Path(__file__).parent.parent.resolve()
SOURCE_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = Path(__file__).parent / "dataset"


def get_images_from_folder(folder_path):
    """Get all image files from a folder"""
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    images = []
    if folder_path.exists():
        for ext in extensions:
            images.extend(folder_path.glob(f'*{ext}'))
            # Don't add uppercase separately - glob is case-insensitive on Windows
    return sorted(set(images))  # Use set to remove any duplicates


def copy_images(image_list, dest_folder):
    """Copy images to destination folder"""
    dest_folder.mkdir(parents=True, exist_ok=True)
    copied = 0
    for img_path in image_list:
        dest_path = dest_folder / img_path.name
        # Handle duplicates by adding suffix
        if dest_path.exists():
            stem = img_path.stem
            suffix = img_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = dest_folder / f"{stem}_{counter}{suffix}"
                counter += 1
        shutil.copy2(img_path, dest_path)
        copied += 1
    return copied


def create_cae_dataset():
    """Create CAE dataset preserving original splits"""
    
    print("=" * 60)
    print("CAE Dataset Creator - PRESERVING ORIGINAL SPLITS")
    print("=" * 60)
    
    # Clean output directory
    if OUTPUT_DIR.exists():
        print(f"\nRemoving existing dataset at {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    print(f"\nCreating CAE dataset at {OUTPUT_DIR}")
    print("\nPreserving original CNN dataset splits to prevent data leakage!")
    
    stats = defaultdict(lambda: defaultdict(int))
    
    # Process each original split separately
    for split in ['training', 'validation', 'testing']:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*60}")
        
        source_split_dir = SOURCE_DIR / split
        if not source_split_dir.exists():
            print(f"Warning: {source_split_dir} not found, skipping")
            continue
        
        # === NORMAL IMAGES (from ND folder) ===
        nd_source = source_split_dir / 'ND'
        if nd_source.exists():
            normal_images = get_images_from_folder(nd_source)
            
            if split == 'training':
                # Training: ALL normal images go to training/normal
                dest = OUTPUT_DIR / 'training' / 'normal'
                n_copied = copy_images(normal_images, dest)
                stats['training']['normal'] = n_copied
                print(f"  Normal -> training/normal: {n_copied} images")
                
            elif split == 'validation':
                # Validation: normal images go to validation/normal
                dest = OUTPUT_DIR / 'validation' / 'normal'
                n_copied = copy_images(normal_images, dest)
                stats['validation']['normal'] = n_copied
                print(f"  Normal -> validation/normal: {n_copied} images")
                
            elif split == 'testing':
                # Testing: normal images go to testing/normal
                dest = OUTPUT_DIR / 'testing' / 'normal'
                n_copied = copy_images(normal_images, dest)
                stats['testing']['normal'] = n_copied
                print(f"  Normal -> testing/normal: {n_copied} images")
        
        # === DEFECT IMAGES (CR, LP, PO) - Only for validation and testing ===
        if split in ['validation', 'testing']:
            for defect_type in ['CR', 'LP', 'PO']:
                defect_source = source_split_dir / defect_type
                if defect_source.exists():
                    defect_images = get_images_from_folder(defect_source)
                    dest = OUTPUT_DIR / split / 'defect' / defect_type
                    n_copied = copy_images(defect_images, dest)
                    stats[split][defect_type] = n_copied
                    print(f"  {defect_type} -> {split}/defect/{defect_type}: {n_copied} images")
    
    # Summary
    print("\n" + "=" * 60)
    print("Dataset Creation Complete!")
    print("=" * 60)
    
    print("\nFinal Statistics:")
    print("-" * 60)
    print(f"{'Split':<15} {'Normal':<10} {'CR':<10} {'LP':<10} {'PO':<10} {'Total':<10}")
    print("-" * 60)
    
    for split in ['training', 'validation', 'testing']:
        normal = stats[split]['normal']
        cr = stats[split].get('CR', 0)
        lp = stats[split].get('LP', 0)
        po = stats[split].get('PO', 0)
        total = normal + cr + lp + po
        print(f"{split:<15} {normal:<10} {cr:<10} {lp:<10} {po:<10} {total:<10}")
    
    print("-" * 60)
    grand_total = sum(sum(v.values()) for v in stats.values())
    print(f"{'TOTAL':<15} {'':<10} {'':<10} {'':<10} {'':<10} {grand_total:<10}")
    
    print(f"\nDataset saved to: {OUTPUT_DIR}")
    print("\nDirectory Structure:")
    print("  cae/dataset/")
    print("  ├── training/")
    print("  │   └── normal/          # ND images from CNN training split")
    print("  ├── validation/")
    print("  │   ├── normal/          # ND images from CNN validation split")
    print("  │   └── defect/")
    print("  │       ├── CR/          # CR images from CNN validation split")
    print("  │       ├── LP/          # LP images from CNN validation split")
    print("  │       └── PO/          # PO images from CNN validation split")
    print("  └── testing/")
    print("      ├── normal/          # ND images from CNN testing split")
    print("      └── defect/")
    print("          ├── CR/          # CR images from CNN testing split")
    print("          ├── LP/          # LP images from CNN testing split")
    print("          └── PO/          # PO images from CNN testing split")
    
    print("\n✓ Original splits preserved - NO DATA LEAKAGE!")


if __name__ == "__main__":
    create_cae_dataset()
