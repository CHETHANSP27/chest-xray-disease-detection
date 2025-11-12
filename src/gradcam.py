"""
Grad-CAM implementation for chest X-ray disease detection
"""

import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class ChestXrayGradCAM:
    def __init__(self, model):
        """
        Initialize Grad-CAM for chest X-ray model
        
        Args:
            model: Trained PyTorch model
        """
        self.model = model
        
        # Ensure model is in eval mode and on CPU
        self.model.eval()
        self.model = self.model.to('cpu')
        
        # Get the target layer (last convolutional layer)
        target_layers = self._get_target_layers(model)
        
        print(f"GradCAM initialized with target layers: {target_layers}")
        
        # Initialize GradCAM - FORCE use_cuda=False for Streamlit Cloud
        self.cam = GradCAM(
            model=model,
            target_layers=target_layers,
            use_cuda=False  # Always use CPU for Streamlit Cloud compatibility
        )
    
    def _get_target_layers(self, model):
        """
        Get the appropriate target layer for the model
        
        Args:
            model: PyTorch model
            
        Returns:
            List of target layers
        """
        # For DenseNet models
        if hasattr(model, 'backbone'):
            if hasattr(model.backbone, 'features'):
                # DenseNet architecture
                return [model.backbone.features[-1]]
        
        # Direct DenseNet model
        if hasattr(model, 'features'):
            return [model.features[-1]]
        
        # ResNet architecture
        if hasattr(model, 'layer4'):
            return [model.layer4[-1]]
        
        # Fallback: use the last convolutional layer
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                return [module]
        
        # Last resort: second to last child
        return [list(model.children())[-2]]
    
    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (None for highest scoring class)
            
        Returns:
            numpy array: Heatmap
        """
        # Ensure input is on CPU
        input_tensor = input_tensor.to('cpu')
        
        # Create targets
        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]
        else:
            targets = None
        
        # Generate CAM
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        
        # Return first image in batch
        return grayscale_cam[0, :]
    
    def visualize(self, input_tensor, original_image, target_class=None):
        """
        Generate and overlay Grad-CAM on original image
        
        Args:
            input_tensor: Input image tensor
            original_image: Original image as numpy array (H, W, 3) in [0, 1]
            target_class: Target class index
            
        Returns:
            numpy array: Visualization image
        """
        # Generate heatmap
        heatmap = self.generate_heatmap(input_tensor, target_class)
        
        # Overlay on image
        visualization = show_cam_on_image(original_image, heatmap, use_rgb=True)
        
        return visualization


def prepare_image_for_cam(tensor):
    """
    Prepare image tensor for CAM visualization
    
    Args:
        tensor: Image tensor (1, C, H, W)
        
    Returns:
        numpy array: Image in [0, 1] range for visualization
    """
    # Convert to numpy and transpose
    image = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    return image