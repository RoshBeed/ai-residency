import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import local modules
from models.classifier.classifier import VisionTransformer
from datasets.mnist_dataset import MNISTDataset

class Trainer:
    def __init__(self, config: Dict):
        """
        Initialize the trainer for encoder classifier.
        
        Args:
            config: Configuration dictionary containing model and training parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.get('seed', 42))
        np.random.seed(config.get('seed', 42))
        
        # Create output directory
        self.output_dir = Path(config.get('output_dir', 'model_outputs'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize model, dataset, and training components
        self._setup_model()
        self._setup_dataset()
        self._setup_training_components()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def _setup_model(self):
        """Initialize the Vision Transformer model."""
        print("Initializing Vision Transformer model...")
        self.model = VisionTransformer(self.config['model_config'])
        self.model.to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    def _setup_dataset(self):
        """Initialize the dataset and data loaders."""
        print("Setting up dataset...")
        dataset_config = self.config['dataset_config']
        
        self.dataset = MNISTDataset(
            data_dir=dataset_config.get('data_dir', 'data'),
            image_size=dataset_config.get('image_size', 28),
            train_split=dataset_config.get('train_split', 0.8),
            batch_size=dataset_config.get('batch_size', 64),
            num_workers=dataset_config.get('num_workers', 4),
            download=dataset_config.get('download', True)
        )
        
        self.train_loader, self.val_loader, self.test_loader = self.dataset.get_data_loaders()
        self.train_size, self.val_size, self.test_size = self.dataset.get_dataset_sizes()
        
        print(f"Dataset sizes - Train: {self.train_size}, Val: {self.val_size}, Test: {self.test_size}")
        
    def _setup_training_components(self):
        """Initialize optimizer, scheduler, and loss function."""
        training_config = self.config['training_config']
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        optimizer_name = training_config.get('optimizer', 'adam').lower()
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=training_config.get('learning_rate', 1e-4),
                weight_decay=training_config.get('weight_decay', 1e-4)
            )
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=training_config.get('learning_rate', 1e-4),
                weight_decay=training_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Learning rate scheduler
        scheduler_name = training_config.get('scheduler', 'cosine').lower()
        if scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config.get('epochs', 100)
            )
        elif scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=training_config.get('step_size', 30),
                gamma=training_config.get('gamma', 0.1)
            )
        else:
            self.scheduler = None
            
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            if self.config['training_config'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training_config']['gradient_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Accuracy: {100. * correct / total:.2f}%')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def validate(self, loader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        # Create descriptive filename with model architecture details
        model_config = self.config['model_config']
        arch_info = f"vit_cls_{model_config['img_size']}x{model_config['img_size']}_patch{model_config['patch_size']}_d{model_config['d_model']}_layers{model_config['num_encoder_layers']}_heads{model_config['heads']}"
        
        # Save regular checkpoint with epoch info
        checkpoint_path = self.output_dir / f'{arch_info}_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model with descriptive name
        if is_best:
            best_path = self.output_dir / f'{arch_info}_best.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation accuracy: {self.val_accuracies[-1]:.2f}%")
            print(f"Model saved as: {best_path}")
    
    def plot_training_history(self):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        epochs = self.config['training_config']['epochs']
        best_val_accuracy = 0.0
        patience = self.config['training_config'].get('patience', 10)
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_accuracy = self.train_epoch()
            
            # Validation
            val_loss, val_accuracy = self.validate(self.val_loader)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f'\nEpoch {epoch}/{epochs} ({epoch_time:.2f}s):')
            print(f'  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
            print(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save checkpoint
            is_best = val_accuracy > best_val_accuracy
            if is_best:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {patience} epochs without improvement')
                break
        
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time:.2f} seconds')
        print(f'Best validation accuracy: {best_val_accuracy:.2f}%')
        
        # Final evaluation on test set
        print("\nEvaluating on test set...")
        test_loss, test_accuracy = self.validate(self.test_loader)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
        
        # Plot training history
        self.plot_training_history()
        
        # Save final model and config
        self.save_checkpoint(epochs, is_best=True)
        
        # Save training config
        with open(self.output_dir / 'training_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        return {
            'best_val_accuracy': best_val_accuracy,
            'test_accuracy': test_accuracy,
            'total_training_time': total_time
        }

def get_default_config():
    """Get default configuration for training."""
    return {
        'model_config': {
            'img_size': 28,
            'patch_size': 4,
            'in_channels': 1,
            'num_classes': 10,
            'd_model': 128,
            'num_encoder_layers': 6,
            'heads': 8,
            'head_dim': 16,
            'ff_hidden_dim': 512,
            'dropout': 0.1
        },
        'dataset_config': {
            'data_dir': 'data',
            'image_size': 28,
            'train_split': 0.8,
            'batch_size': 64,
            'num_workers': 4,
            'download': True
        },
        'training_config': {
            'epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'gradient_clip': 1.0,
            'patience': 15
        },
        'output_dir': 'model_outputs',
        'seed': 42
    }

def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description='Train Vision Transformer Encoder Classifier')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--output_dir', type=str, default='model_outputs', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of encoder layers')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    config['output_dir'] = args.output_dir
    config['training_config']['epochs'] = args.epochs
    config['training_config']['learning_rate'] = args.learning_rate
    config['dataset_config']['batch_size'] = args.batch_size
    config['model_config']['d_model'] = args.d_model
    config['model_config']['num_encoder_layers'] = args.num_layers
    
    # Create trainer and start training
    trainer = Trainer(config)
    results = trainer.train()
    
    print(f"\nTraining Results:")
    print(f"Best Validation Accuracy: {results['best_val_accuracy']:.2f}%")
    print(f"Test Accuracy: {results['test_accuracy']:.2f}%")
    print(f"Total Training Time: {results['total_training_time']:.2f} seconds")

if __name__ == '__main__':
    main() 