"""
Grad-CAM implementation for model explainability
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class ChestXrayGradCAM:
    """Grad-CAM wrapper for chest X-ray model"""
    
    def __init__(self, model, target_layers=None):
        self.model = model
        self.model.eval()
        
        # Set target layer for Grad-CAM
        if target_layers is None:
            # Use last convolutional layer of DenseNet
            target_layers = [model.backbone.features[-1]]
        
        self.cam = GradCAM(
            model=model,
            target_layers=target_layers,
            use_cuda=torch.cuda.is_available()
        )
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, 3, H, W)
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            cam: Grad-CAM heatmap
            prediction: Model prediction
        """
        # Get prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        # If no target class specified, use the class with highest probability
        if target_class is None:
            target_class = torch.argmax(prediction, dim=1).item()
        
        # Create target for Grad-CAM
        targets = [ClassifierOutputTarget(target_class)]
        
        # Generate CAM
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        return grayscale_cam, prediction.cpu().numpy()[0]
    
    def visualize_cam(self, image, cam, alpha=0.5):
        """
        Overlay CAM on image
        
        Args:
            image: Original image (H, W, 3) in [0, 1] range
            cam: Grad-CAM heatmap
            alpha: Transparency of overlay
            
        Returns:
            visualization: Image with CAM overlay
        """
        visualization = show_cam_on_image(image, cam, use_rgb=True)
        return visualization
    
    def generate_multi_class_cams(self, input_tensor, image, disease_labels, top_k=3):
        """
        Generate CAMs for multiple diseases
        
        Args:
            input_tensor: Input image tensor
            image: Original image (normalized)
            disease_labels: List of disease names
            top_k: Number of top diseases to visualize
            
        Returns:
            visualizations: Dictionary of disease -> CAM visualization
            predictions: Dictionary of disease -> probability
        """
        # Get predictions
        with torch.no_grad():
            predictions = self.model(input_tensor).cpu().numpy()[0]
        
        # Get top-k diseases
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        visualizations = {}
        prediction_dict = {}
        
        for idx in top_indices:
            disease = disease_labels[idx]
            prob = predictions[idx]
            
            # Generate CAM for this disease
            targets = [ClassifierOutputTarget(idx)]
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Create visualization
            viz = show_cam_on_image(image, grayscale_cam, use_rgb=True)
            
            visualizations[disease] = viz
            prediction_dict[disease] = prob
        
        return visualizations, prediction_dict

def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image tensor for visualization"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def prepare_image_for_cam(image_tensor):
    """Prepare image tensor for CAM visualization"""
    # Denormalize
    image = denormalize_image(image_tensor[0])
    
    # Convert to numpy
    image = image.permute(1, 2, 0).cpu().numpy()
    
    return image

# Example usage
if __name__ == "__main__":
    from src.model import create_model
    from src.config import Config
    
    # Create model
    model = create_model()
    
    # Create Grad-CAM
    gradcam = ChestXrayGradCAM(model)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Generate CAM
    cam, prediction = gradcam.generate_cam(dummy_input)
    
    print(f"CAM shape: {cam.shape}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Top prediction: {np.argmax(prediction)}")
