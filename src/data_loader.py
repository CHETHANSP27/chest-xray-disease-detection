"""
Data loading and preprocessing for chest X-ray images
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from src.config import Config

class ChestXrayDataset(Dataset):
    """Custom Dataset for Chest X-ray images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Get labels
        label = torch.FloatTensor(self.labels[idx])
        
        return image, label, img_path

def get_transforms(mode='train'):
    """Get image transformations for training/validation/test"""
    
    if mode == 'train':
        return A.Compose([
            A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            A.HorizontalFlip(p=Config.AUGMENTATION['horizontal_flip_prob']),
            A.Rotate(limit=Config.AUGMENTATION['rotation_limit'], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=Config.AUGMENTATION['brightness_limit'],
                contrast_limit=Config.AUGMENTATION['contrast_limit'],
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

def prepare_data(csv_path, images_dir):
    """Prepare dataset from CSV file"""
    
    print("Loading dataset metadata...")
    df = pd.read_csv(csv_path)
    
    # Create binary labels for each disease
    labels = []
    image_paths = []
    
    for idx, row in df.iterrows():
        img_name = row['Image Index']
        img_path = os.path.join(images_dir, img_name)
        
        # Check if image exists
        if not os.path.exists(img_path):
            continue
            
        # Parse labels
        finding_labels = row['Finding Labels'].split('|')
        
        # Create binary vector for 14 diseases
        label_vector = [0] * Config.NUM_CLASSES
        for label in finding_labels:
            if label in Config.DISEASE_LABELS:
                idx_label = Config.DISEASE_LABELS.index(label)
                label_vector[idx_label] = 1
        
        labels.append(label_vector)
        image_paths.append(img_path)
    
    labels = np.array(labels)
    
    print(f"Total images loaded: {len(image_paths)}")
    print(f"Label distribution:")
    for i, disease in enumerate(Config.DISEASE_LABELS):
        print(f"  {disease}: {labels[:, i].sum()} ({labels[:, i].mean()*100:.2f}%)")
    
    return image_paths, labels

def create_data_loaders(csv_path, images_dir, batch_size=None):
    """Create train, validation, and test data loaders"""
    
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    
    # Prepare data
    image_paths, labels = prepare_data(csv_path, images_dir)
    
    # Split data: train, val, test
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, 
        test_size=(Config.VAL_SPLIT + Config.TEST_SPLIT),
        random_state=42,
        stratify=labels[:, 0]  # Stratify by first disease label
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=Config.TEST_SPLIT/(Config.VAL_SPLIT + Config.TEST_SPLIT),
        random_state=42,
        stratify=temp_labels[:, 0]
    )
    
    print(f"\nDataset splits:")
    print(f"  Training: {len(train_paths)} images")
    print(f"  Validation: {len(val_paths)} images")
    print(f"  Testing: {len(test_paths)} images")
    
    # Create datasets
    train_dataset = ChestXrayDataset(
        train_paths, train_labels, 
        transform=get_transforms('train')
    )
    val_dataset = ChestXrayDataset(
        val_paths, val_labels,
        transform=get_transforms('val')
    )
    test_dataset = ChestXrayDataset(
        test_paths, test_labels,
        transform=get_transforms('test')
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# Utility function for single image loading (for Streamlit app)
def load_single_image(image_path):
    """Load and preprocess a single image"""
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    transform = get_transforms('test')
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0)
    
    return image_tensor
