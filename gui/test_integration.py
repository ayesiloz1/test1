"""
Test CAE integration with GUI
"""

from config import GUIConfig
from inference_engine import DefectDetectionEngine
from pathlib import Path

# Test configuration
config = GUIConfig()
print(f"CNN Model: {config.cnn_model_path}")
print(f"CAE Model: {config.autoencoder_model_path}")
print(f"CAE Threshold: {config.ae_threshold}")
print(f"Use CAE: {config.use_autoencoder}")

# Initialize engine
print("\nInitializing detection engine...")
engine = DefectDetectionEngine(config)

# Test with a sample image
test_image = None
test_dir = Path(config.project_root) / 'dataset' / 'testing'
if test_dir.exists():
    for class_dir in ['CR', 'LP', 'ND', 'PO']:
        test_path = test_dir / class_dir
        if test_path.exists():
            images = list(test_path.glob('*.jpg')) + list(test_path.glob('*.png'))
            if images:
                test_image = images[0]
                break

if not test_image:
    # Try backup CNN dataset
    test_dir = Path(config.project_root) / 'backup_cnn' / 'dataset' / 'testing'
    if test_dir.exists():
        for class_dir in ['CR', 'LP', 'ND', 'PO']:
            test_path = test_dir / class_dir
            if test_path.exists():
                images = list(test_path.glob('*.jpg')) + list(test_path.glob('*.png'))
                if images:
                    test_image = images[0]
                    break

if test_image and test_image.exists():
    print(f"\nTesting with: {test_image}")
    result = engine.predict(str(test_image), config)
    
    print("\n=== Results ===")
    if result['cnn']:
        cnn = result['cnn']
        print(f"CNN Prediction: {cnn['prediction']} ({cnn['confidence']:.2%})")
        print(f"  Probabilities: {', '.join([f'{k}:{v:.2%}' for k, v in cnn['probabilities'].items()])}")
    
    if result['autoencoder']:
        ae = result['autoencoder']
        print(f"\nCAE Error: {ae['error']:.6f}")
        print(f"CAE Threshold: {ae['threshold']:.6f}")
        print(f"CAE Status: {ae['status']}")
        print(f"Heatmap shape: {ae['heatmap'].shape}")
    
    print(f"\nProcessing time: {result['processing_time']:.3f}s")
else:
    print(f"\nNo test images found. Please check dataset paths.")
