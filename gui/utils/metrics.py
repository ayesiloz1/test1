"""
Metrics and Confusion Matrix Utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support


def create_confusion_matrix_figure(y_true, y_pred, class_names, title="Confusion Matrix"):
    """
    Create a matplotlib figure with confusion matrix
    
    Args:
        y_true: List of ground truth labels
        y_pred: List of predicted labels
        class_names: List of class names ['CR', 'LP', 'ND', 'PO']
        title: Title for the plot
        
    Returns:
        matplotlib Figure object
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    # Create figure
    fig = Figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=12, fontweight='bold')
    
    fig.tight_layout()
    return fig


def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculate accuracy, precision, recall, and F1 score
    
    Returns:
        dict with metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def get_hybrid_prediction(cnn_pred, cnn_conf, ae_is_anomaly, confidence_threshold=0.49):
    """
    Get hybrid prediction combining CNN and AE results
    
    Args:
        cnn_pred: CNN predicted class
        cnn_conf: CNN confidence
        ae_is_anomaly: Boolean, True if AE detected anomaly
        confidence_threshold: Threshold for trusting CNN (default 0.49)
        
    Returns:
        str: Final prediction class
    """
    # If CNN is confident (>49%), trust CNN classification
    if cnn_conf > confidence_threshold:
        return cnn_pred
    else:
        # CNN uncertain, use AE result
        if ae_is_anomaly:
            # Detected as anomaly but uncertain which type, use CNN's best guess
            return cnn_pred
        else:
            # No anomaly detected
            return 'ND'
