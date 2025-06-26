import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import time
from tqdm import tqdm
import sys
sys.path.append('..')

from transformer import Seq2SeqVisionTransformer
from datasets import MNISTDataset
from config import MODEL_CONFIG, TRAINING_CONFIG, DATASET_CONFIG, MODEL_PATHS

def create_single_digit_config():
    """Create config for single digit Seq2SeqVisionTransformer"""
    config = MODEL_CONFIG.copy()
    config["img_size"] = 28  # MNIST image size
    config["patch_size"] = 4  # Smaller patches for MNIST
    config["max_seq_length"] = 2  # [digit, EOS]
    config["vocab_size"] = 12  # 0-9 digits + EOS token (1) + PAD token (0)
    return config

def create_single_digit_sequences(labels):
    """Convert single digit labels to sequences with EOS token"""
    sequences = []
    for label in labels:
        # Create sequence: [digit, EOS]
        sequence = [label.item(), 1]  # 1 is EOS token
        sequences.append(sequence)
    return torch.tensor(sequences, dtype=torch.long)

def sequence_to_single_digit(sequence):
    """Convert sequence back to single digit"""
    # Take the first token (before EOS)
    return sequence[0].item()

def train_single_digit_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=10, device='cpu'):
    """
    Train the Seq2Seq VisionTransformer model for single digit recognition.
    
    Args:
        model: Seq2SeqVisionTransformer model
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
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
        
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Create target sequences
            target_sequences = create_single_digit_sequences(labels).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, target_sequences)
            
            # Calculate loss for each position
            loss = 0
            seq_len = outputs.size(1)
            
            for pos in range(seq_len):
                pos_output = outputs[:, pos, :]
                pos_target = target_sequences[:, pos]
                pos_loss = criterion(pos_output, pos_target)
                loss += pos_loss
            
            # Average loss over sequence length
            loss = loss / seq_len
            
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
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device)
                
                # Create target sequences for validation
                target_sequences = create_single_digit_sequences(labels).to(device)
                
                # Forward pass
                outputs = model(images, target_sequences)
                
                # Convert predictions to sequences
                predicted_sequences = torch.argmax(outputs, dim=-1)  # (batch_size, seq_len)
                
                # Convert sequences back to digits for comparison
                for i in range(len(predicted_sequences)):
                    pred_digit = sequence_to_single_digit(predicted_sequences[i])
                    target_digit = labels[i].item()
                    
                    if pred_digit == target_digit:
                        correct += 1
                    total += 1
                
                # Update validation progress bar
                current_acc = 100 * correct / total
                val_pbar.set_postfix({'Val Acc': f'{current_acc:.1f}%'})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s')
        print(f'Training Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_acc:.2f}%')
        
        # Step the scheduler based on validation accuracy
        scheduler.step(epoch_acc)
        current_lr = scheduler.get_last_lr()[0]
        print(f'Learning Rate: {current_lr:.6f}')
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "model_data/best_single_digit_vit.pth")
            print(f'New best model saved with accuracy: {best_acc:.2f}%')
        
        print('-' * 50)
    
    return best_acc

def evaluate_single_digit_model(model, test_loader, device='cpu'):
    """
    Evaluate the trained single digit model.
    
    Args:
        model: Trained Seq2SeqVisionTransformer model
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
            
            # Create target sequences
            target_sequences = create_single_digit_sequences(labels).to(device)
            
            outputs = model(images, target_sequences)
            predicted_sequences = torch.argmax(outputs, dim=-1)
            
            # Convert sequences back to digits for comparison
            for i in range(len(predicted_sequences)):
                pred_digit = sequence_to_single_digit(predicted_sequences[i])
                target_digit = labels[i].item()
                
                if pred_digit == target_digit:
                    correct += 1
                total += 1
    
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
    
    # Create dataset using the new MNISTDataset
    dataset = MNISTDataset(
        data_dir="data",
        image_size=28,
        train_split=0.8,
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
    config = create_single_digit_config()
    model = Seq2SeqVisionTransformer(config)
    
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
    best_accuracy = train_single_digit_model(
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
    torch.save(model.state_dict(), "model_data/final_single_digit_vit.pth")
    print("Final model saved to model_data/final_single_digit_vit.pth")
    
    # Evaluate on test set
    print("\nEvaluating final model on test set...")
    evaluate_single_digit_model(model, test_loader, device=device) 