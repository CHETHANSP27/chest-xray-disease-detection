"""
Utility functions for the project
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import json
import os

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Only load optimizer state if optimizer is provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {filepath} (Epoch {epoch})")
    return model, optimizer, epoch, loss

def plot_training_history(history, save_path):
    """Plot training and validation loss/metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot AUC
    axes[1].plot(history['train_auc'], label='Train AUC')
    axes[1].plot(history['val_auc'], label='Validation AUC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC Score')
    axes[1].set_title('Training and Validation AUC')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved: {save_path}")
    plt.close()

def plot_roc_curves(y_true, y_pred, class_names, save_path):
    """Plot ROC curves for all classes"""
    n_classes = len(class_names)
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.ravel()
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        
        axes[i].plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        axes[i].plot([0, 1], [0, 1], 'k--')
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title(class_names[i])
        axes[i].legend()
        axes[i].grid(True)
    
    # Hide extra subplots
    for i in range(n_classes, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved: {save_path}")
    plt.close()

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calculate classification metrics"""
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Calculate AUC for each class
    auc_scores = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:  # Only if both classes present
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            auc_scores.append(auc)
    
    mean_auc = np.mean(auc_scores) if auc_scores else 0.0
    
    return {
        'mean_auc': mean_auc,
        'individual_auc': auc_scores
    }

def save_predictions(predictions, labels, image_paths, save_path):
    """Save predictions to JSON file"""
    results = {
        'predictions': predictions.tolist(),
        'labels': labels.tolist(),
        'image_paths': image_paths
    }
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Predictions saved: {save_path}")

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=5, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        
        return self.early_stop
