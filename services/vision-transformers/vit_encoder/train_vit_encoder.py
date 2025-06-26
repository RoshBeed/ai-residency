import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from tqdm import tqdm
import sys
sys.path.append('..')

from transformer import VisionTransformer
from datasets import MultiDigitMNISTDataset
from config import MODEL_CONFIG, TRAINING_CONFIG, DATASET_CONFIG, MODEL_PATHS

def create_vit_config():
    """Create config for VisionTransformer from global config"""
    config = MODEL_CONFIG.copy()
    config["num_classes"] = 1000  # 0-999 for multi-digit numbers
    return config

def train_vit_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=10, device='cpu'):
    """
    Train the VisionTransformer model for classification.
    
    Args:
        model: VisionTransformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        scheduler: Learning rate scheduler
        epochs: Number of training epochs
        device: Device to train on
    """
    model.to(device)
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        # Progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
        
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        # Progress bar for validation batches
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_pbar):
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Get predictions
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                current_acc = 100 * correct / total
                val_pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s')
        print(f'Training Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_acc:.2f}%')
        
        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f'Learning Rate: {current_lr:.6f}')
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), MODEL_PATHS["best_model"])
            print(f'New best model saved with accuracy: {best_acc:.2f}%')
        
        print('-' * 50)
    
    return best_acc

def evaluate_vit_model(model, test_loader, device='cpu'):
    """
    Evaluate the trained VisionTransformer model.
    
    Args:
        model: Trained VisionTransformer model
        test_loader: Test data loader
        device: Device to evaluate on
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating on test set'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

if __name__ == "__main__":
    # Set device
    device = TRAINING_CONFIG["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
        print("MPS not available, using CPU")
    
    print(f"Using device: {device}")
    
    # Create dataset using the new MultiDigitMNISTDataset
    dataset = MultiDigitMNISTDataset(
        data_dir="data",
        image_size=DATASET_CONFIG["image_size"],
        max_digits=DATASET_CONFIG["max_digits"],
        train_split=DATASET_CONFIG["train_split"],
        samples_per_class=10,  # Balanced dataset
        batch_size=TRAINING_CONFIG["batch_size"],
        num_workers=4,
        download=True
    )
    
    # Get data loaders
    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    
    # Print dataset information
    train_size, val_size, test_size = dataset.get_dataset_sizes()
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Test samples: {test_size}")
    
    # Create model
    config = create_vit_config()
    model = VisionTransformer(config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"]
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Train model
    print("Starting training...")
    best_accuracy = train_vit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=TRAINING_CONFIG["epochs"],
        device=device
    )
    
    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), MODEL_PATHS["final_model"])
    print(f"Final model saved to {MODEL_PATHS['final_model']}")
    
    # Evaluate on test set
    print("\nEvaluating final model on test set...")
    evaluate_vit_model(model, test_loader, device=device) 