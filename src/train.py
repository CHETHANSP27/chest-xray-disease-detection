"""
Training script for chest X-ray disease detection model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from src.config import Config
from src.model import create_model
from src.data_loader import create_data_loaders
from src.utils import save_checkpoint, plot_training_history, EarlyStopping, calculate_metrics
import torch
import torch.nn as nn
import torch.cuda as cuda


class Trainer:
    """Trainer class for model training"""
    
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        
        
        # Loss function (Binary Cross Entropy for multi-label)
        self.criterion = nn.BCELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=Config.PATIENCE,
            mode='min'
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': []
        }

    def save_state(self, epoch, val_loss):
        """Save complete training state"""
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history
        }
        
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
        torch.save(state, checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path}")

    def load_state(self, checkpoint_path):
        """Load training state from checkpoint"""
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path)
            self.model.load_state_dict(state['model_state'])
            self.optimizer.load_state_dict(state['optimizer_state'])
            self.start_epoch = state['epoch']
            self.best_val_loss = state['best_val_loss']
            self.train_history = state.get('train_history', [])
            print(f"Resumed from epoch {self.start_epoch}")
            return True
        return False
    
    def _get_latest_checkpoint(self):
        """Get the most recent checkpoint file"""
        if not os.path.exists(Config.CHECKPOINT_DIR):
            return None
        checkpoints = [f for f in os.listdir(Config.CHECKPOINT_DIR) if f.startswith('checkpoint_epoch_')]
        if not checkpoints:
            return None
        latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return os.path.join(Config.CHECKPOINT_DIR, latest)

    def save_progress(self, epoch, val_loss):
        """Save training progress"""
        progress = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'early_stopping_counter': self.early_stopping.counter,
            'early_stopping_best_score': self.early_stopping.best_score
        }
        
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
        torch.save(progress, checkpoint_path)
        print(f"\nProgress saved: {checkpoint_path}")
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
         # Add memory clearance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            all_preds.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        epoch_loss = running_loss / len(self.train_loader.dataset)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        metrics = calculate_metrics(all_labels, all_preds)
        
        return epoch_loss, metrics['mean_auc']
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, labels, _ in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        epoch_loss = running_loss / len(self.val_loader.dataset)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        metrics = calculate_metrics(all_labels, all_preds)
        
        return epoch_loss, metrics['mean_auc']
    
    def train(self, num_epochs):
        """Train with support for incremental runs"""
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.MODEL_DIR, exist_ok=True)

        # Load previous progress if exists
        latest_checkpoint = self._get_latest_checkpoint()
        if latest_checkpoint:
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            start_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.history = checkpoint['history']
            self.early_stopping.counter = checkpoint['early_stopping_counter']
            self.early_stopping.best_score = checkpoint['early_stopping_best_score']
        else:
            start_epoch = 0

        print(f"\nStarting training from epoch {start_epoch + 1}")
        print(f"Training until epoch {min(start_epoch + Config.EPOCHS_PER_SESSION, num_epochs)}")
        print(f"Device: {self.device}")

        for epoch in range(start_epoch, min(start_epoch + Config.EPOCHS_PER_SESSION, num_epochs)):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)

            # Train and validate
            train_loss, train_auc = self.train_epoch()
            val_loss, val_auc = self.validate()

            # Update history and print metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            
            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Save progress after each epoch
            self.save_progress(epoch + 1, val_loss)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_path = os.path.join(Config.MODEL_DIR, 'best_model.pth')
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, best_model_path)
                print(f"New best model saved! (Val Loss: {val_loss:.4f})")

            # Early stopping check
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        remaining_epochs = num_epochs - (start_epoch + Config.EPOCHS_PER_SESSION)
        if remaining_epochs > 0:
            print(f"\nTraining paused. {remaining_epochs} epochs remaining.")
            print("To continue training, run the script again.")
        else:
            print("\nTraining completed!")
            plot_path = os.path.join(Config.MODEL_DIR, 'training_history.png')
            plot_training_history(self.history, plot_path)

def check_gpu():
    """Check GPU availability and properties"""
    if not cuda.is_available():
        print("WARNING: CUDA is not available. Running on CPU!")
        return
    
    print("\nGPU Information:")
    print(f"CUDA Available: {cuda.is_available()}")
    print(f"Current Device: {cuda.current_device()}")
    print(f"Device Name: {cuda.get_device_name()}")
    print(f"Device Count: {cuda.device_count()}")
    print(f"Memory Usage:")
    print(f"  Allocated: {round(cuda.memory_allocated(0)/1024**2,1)} MB")
    print(f"  Cached: {round(cuda.memory_reserved(0)/1024**2,1)} MB")

def main():
    # Set GPU memory usage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True  # This can speed up training
        torch.backends.cudnn.deterministic = False
    """Main training function"""
    print("=" * 60)
    print("CHEST X-RAY DISEASE DETECTION - TRAINING")
    print("=" * 60)
        # Check GPU
    check_gpu()
    
    # Set paths (UPDATE THESE PATHS BASED ON YOUR DATA LOCATION)
    csv_path = os.path.join(Config.METADATA_DIR, 'Data_Entry_2017.csv')
    images_dir = os.path.join(Config.RAW_DATA_DIR, 'images')
    
    # Create data loaders
    print("\nPreparing data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        csv_path,
        images_dir
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, Config.DEVICE)
    
    # Train model
    history = trainer.train(Config.NUM_EPOCHS)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Best model saved at: {os.path.join(Config.MODEL_DIR, 'best_model.pth')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
