"""
Configuration file for chest X-ray disease detection project
Contains all hyperparameters and settings
"""

import torch
import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    METADATA_DIR = os.path.join(DATA_DIR, 'metadata')
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
    
    # Dataset Parameters
    IMAGE_SIZE = 224  # Standard size for DenseNet
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Disease Labels (14 diseases from NIH Chest X-ray14)
    DISEASE_LABELS = [
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
        'Consolidation',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Pleural_Thickening',
        'Hernia'
    ]
    
    NUM_CLASSES = len(DISEASE_LABELS)
    
    # Model Parameters
    MODEL_NAME = 'densenet121'
    PRETRAINED = True
    DROPOUT_RATE = 0.5
    
    # Training Parameters
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 25
    PATIENCE = 5  # Early stopping patience
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Device Configuration
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Data Augmentation Parameters
    AUGMENTATION = {
        'horizontal_flip_prob': 0.5,
        'rotation_limit': 10,
        'brightness_limit': 0.2,
        'contrast_limit': 0.2
    }
    
    # Grad-CAM Parameters
    GRADCAM_LAYER = 'features'  # Layer for Grad-CAM visualization
    
    # Evaluation Thresholds
    CLASSIFICATION_THRESHOLD = 0.5
    
    # Create directories if they don't exist
    @staticmethod
    def create_directories():
        directories = [
            Config.RAW_DATA_DIR,
            Config.PROCESSED_DATA_DIR,
            Config.METADATA_DIR,
            Config.MODEL_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Initialize configuration
Config.create_directories()
