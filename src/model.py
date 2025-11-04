"""
CNN Model Architecture for Chest X-ray Disease Detection
Uses DenseNet121 with custom classification head
"""

import torch
import torch.nn as nn
import torchvision.models as models
from src.config import Config

class ChestXrayModel(nn.Module):
    """
    DenseNet121-based model for multi-label classification
    of chest X-ray diseases
    """
    
    def __init__(self, num_classes=14, pretrained=True, dropout_rate=0.5):
        super(ChestXrayModel, self).__init__()
        
        # Load pretrained DenseNet121
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Get number of features from last layer
        num_features = self.backbone.classifier.in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
            nn.Sigmoid()  # Multi-label classification
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Extract features before classification layer"""
        features = self.backbone.features(x)
        return features

def create_model(num_classes=None, pretrained=None, dropout_rate=None):
    """Factory function to create model with default config"""
    
    if num_classes is None:
        num_classes = Config.NUM_CLASSES
    if pretrained is None:
        pretrained = Config.PRETRAINED
    if dropout_rate is None:
        dropout_rate = Config.DROPOUT_RATE
    
    model = ChestXrayModel(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    
    return model

# Test model architecture
if __name__ == "__main__":
    model = create_model()
    print(model)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (probabilities): {output}")
