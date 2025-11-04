"""
Model evaluation script
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score
from src.config import Config
from src.model import create_model
from src.data_loader import create_data_loaders
from src.utils import load_checkpoint, plot_roc_curves, calculate_metrics

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    all_paths = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_paths.extend(paths)
    
    # Concatenate all predictions
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    return all_preds, all_labels, all_paths

def print_evaluation_results(y_true, y_pred, disease_labels):
    """Print detailed evaluation metrics"""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Calculate overall metrics
    metrics = calculate_metrics(y_true, y_pred)
    print(f"\nOverall Mean AUC: {metrics['mean_auc']:.4f}")
    
    # Per-class metrics
    print("\nPer-Disease Metrics:")
    print("-" * 60)
    print(f"{'Disease':<25} {'AUC':<10} {'Prevalence':<15}")
    print("-" * 60)
    
    for i, disease in enumerate(disease_labels):
        if len(np.unique(y_true[:, i])) > 1:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        else:
            auc = 0.0
        
        prevalence = y_true[:, i].mean() * 100
        print(f"{disease:<25} {auc:<10.4f} {prevalence:<15.2f}%")
    
    print("=" * 60)

def main():
    """Main evaluation function"""
    print("=" * 60)
    print("CHEST X-RAY DISEASE DETECTION - EVALUATION")
    print("=" * 60)
    
    # Set paths
    csv_path = os.path.join(Config.METADATA_DIR, 'Data_Entry_2017.csv')
    images_dir = os.path.join(Config.RAW_DATA_DIR, 'images')
    model_path = os.path.join(Config.MODEL_DIR, 'best_model.pth')
    
    # Create data loaders
    print("\nLoading test data...")
    _, _, test_loader = create_data_loaders(csv_path, images_dir)
    
    # Create model
    print("\nLoading model...")
    model = create_model()
    
    # Load checkpoint
    optimizer = torch.optim.Adam(model.parameters())
    model, _, _, _ = load_checkpoint(model, optimizer, model_path, Config.DEVICE)
    model = model.to(Config.DEVICE)
    
    # Evaluate
    predictions, labels, paths = evaluate_model(model, test_loader, Config.DEVICE)
    
    # Print results
    print_evaluation_results(labels, predictions, Config.DISEASE_LABELS)
    
    # Plot ROC curves
    roc_save_path = os.path.join(Config.MODEL_DIR, 'roc_curves.png')
    plot_roc_curves(labels, predictions, Config.DISEASE_LABELS, roc_save_path)
    
    print(f"\nROC curves saved to: {roc_save_path}")
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()
