"""
Helper script to extract and add metrics from trained models to knowledge base
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from knowledge_base import initialize_default_knowledge_base, MetricsKnowledgeBase


def extract_cnn_metrics():
    """
    Extract metrics from CNN model training
    This is a template - modify based on your actual training script
    """
    # Load from training log or model checkpoint
    model_path = Path(__file__).parent.parent / 'models' / 'best_model_pytorch.pth'
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    
    # Extract metrics if they were saved
    metrics = {}
    
    if isinstance(checkpoint, dict):
        # If checkpoint contains training history
        if 'train_loss' in checkpoint:
            metrics['train_loss'] = checkpoint.get('train_loss', [])
        if 'val_loss' in checkpoint:
            metrics['val_loss'] = checkpoint.get('val_loss', [])
        if 'train_acc' in checkpoint:
            metrics['train_accuracy'] = checkpoint.get('train_acc', [])
        if 'val_acc' in checkpoint:
            metrics['val_accuracy'] = checkpoint.get('val_acc', [])
        if 'epoch' in checkpoint:
            metrics['best_epoch'] = checkpoint.get('epoch', 0)
    
    return metrics


def create_confusion_matrix_from_predictions(predictions_file=None):
    """
    Create confusion matrix from saved predictions
    
    Args:
        predictions_file: Path to JSON file with predictions and labels
                         Format: {"predictions": [...], "labels": [...]}
    """
    if predictions_file and Path(predictions_file).exists():
        with open(predictions_file, 'r') as f:
            data = json.load(f)
        
        predictions = np.array(data['predictions'])
        labels = np.array(data['labels'])
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, predictions)
        
        return cm
    
    return None


def add_current_metrics():
    """Add metrics from current project to knowledge base"""
    
    print("="*60)
    print("Adding Model Metrics to Knowledge Base")
    print("="*60 + "\n")
    
    # Initialize KB
    kb_path = Path(__file__).parent / "knowledge_base"
    kb = initialize_default_knowledge_base(str(kb_path))
    metrics_kb = MetricsKnowledgeBase(kb)
    
    # Example: Add CNN metrics
    print("1. Adding CNN Classifier Metrics...")
    
    # You can manually create metrics or load from files
    cnn_metrics = {
        "model_type": "CNN Classifier (ResNet18)",
        "training_accuracy": 0.986,
        "validation_accuracy": 0.924,
        "test_accuracy": 0.942,
        "parameters": "~11M",
        "training_time": "2.5 hours",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    }
    
    metrics_kb.add_training_metrics(
        cnn_metrics, 
        metadata={
            "model": "CNN Classifier",
            "date": "2024-02-01",
            "dataset": "Weld Defect Dataset v1.0"
        }
    )
    print("✓ CNN metrics added")
    
    # Example: Add Confusion Matrix
    print("\n2. Adding Confusion Matrix...")
    
    # Example confusion matrix (replace with actual values)
    confusion_matrix = np.array([
        [45, 2, 1, 0],   # CR predictions
        [1, 42, 0, 3],   # LP predictions
        [0, 1, 48, 1],   # ND predictions
        [2, 1, 0, 44]    # PO predictions
    ])
    
    labels = ["CR", "LP", "ND", "PO"]
    
    metrics_kb.add_confusion_matrix(
        confusion_matrix,
        labels,
        metadata={
            "model": "CNN Classifier",
            "date": "2024-02-01",
            "dataset": "Test Set (191 samples)"
        }
    )
    print("✓ Confusion matrix added")
    
    # Example: Add CAE metrics
    print("\n3. Adding Convolutional Autoencoder Metrics...")
    
    cae_metrics = {
        "model_type": "Convolutional Autoencoder",
        "reconstruction_loss": 0.015,
        "optimal_threshold": 0.018,
        "roc_auc": 0.95,
        "parameters": "~2M",
        "training_time": "3 hours",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 20
    }
    
    metrics_kb.add_training_metrics(
        cae_metrics,
        metadata={
            "model": "Convolutional Autoencoder",
            "date": "2024-02-01",
            "dataset": "Normal welds only (unsupervised)"
        }
    )
    print("✓ CAE metrics added")
    
    # Example: Add model comparison
    print("\n4. Adding Model Comparison...")
    
    comparison = {
        "hybrid_approach": "CAE (anomaly detection) + CNN (classification)",
        "cnn_accuracy": 0.942,
        "cae_auc": 0.95,
        "combined_performance": "More robust than either model alone",
        "advantages": [
            "CAE detects novel defects not in training data",
            "CNN provides specific defect classification",
            "Complementary strengths reduce false negatives"
        ]
    }
    
    metrics_kb.add_model_performance(
        comparison,
        metadata={
            "comparison": "Hybrid System",
            "date": "2024-02-01"
        }
    )
    print("✓ Model comparison added")
    
    # Show statistics
    print("\n" + "="*60)
    stats = kb.get_statistics()
    print(f"Knowledge Base Updated!")
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Total Embeddings: {stats['total_embeddings']}")
    print("="*60)


def add_metrics_from_file(json_file):
    """
    Add metrics from a JSON file
    
    JSON format:
    {
        "confusion_matrix": {
            "matrix": [[...], ...],
            "labels": ["CR", "LP", "ND", "PO"]
        },
        "training_metrics": {
            "train_loss": [...],
            "val_loss": [...],
            "train_accuracy": [...],
            "val_accuracy": [...]
        },
        "metadata": {
            "model": "CNN",
            "date": "2024-02-01"
        }
    }
    """
    if not Path(json_file).exists():
        print(f"File not found: {json_file}")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Initialize KB
    kb_path = Path(__file__).parent / "knowledge_base"
    kb = initialize_default_knowledge_base(str(kb_path))
    metrics_kb = MetricsKnowledgeBase(kb)
    
    metadata = data.get('metadata', {})
    
    # Add confusion matrix if present
    if 'confusion_matrix' in data:
        cm_data = data['confusion_matrix']
        cm = np.array(cm_data['matrix'])
        labels = cm_data['labels']
        metrics_kb.add_confusion_matrix(cm, labels, metadata)
        print("✓ Confusion matrix added from file")
    
    # Add training metrics if present
    if 'training_metrics' in data:
        metrics_kb.add_training_metrics(data['training_metrics'], metadata)
        print("✓ Training metrics added from file")
    
    # Show statistics
    stats = kb.get_statistics()
    print(f"\nKnowledge Base Updated!")
    print(f"Total Documents: {stats['total_documents']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Add from file
        json_file = sys.argv[1]
        add_metrics_from_file(json_file)
    else:
        # Add example metrics
        print("Adding example metrics to knowledge base...")
        print("(Modify this script to add your actual model metrics)\n")
        add_current_metrics()
        print("\nTo add metrics from a file:")
        print("  python add_metrics_to_kb.py metrics.json")
