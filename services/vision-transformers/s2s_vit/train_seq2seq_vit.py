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
from datasets import MultiDigitMNISTDataset
from config import MODEL_CONFIG, TRAINING_CONFIG, DATASET_CONFIG, MODEL_PATHS

def create_seq2seq_config():
    """Create config for Seq2SeqVisionTransformer from global config"""
    config = MODEL_CONFIG.copy()
    config["max_seq_length"] = TRAINING_CONFIG["max_digits"] + 2  # +2 for START and EOS tokens
    return config

def create_digit_sequences(labels, max_digits=3, vocab_size=12):
    """Convert labels to digit sequences with START and EOS tokens, ensuring all values are in [0, vocab_size-1]."""
    sequences = []
    START_TOKEN = 10  # START token
    EOS_TOKEN = 11    # EOS token
    
    for label in labels:
        # Ensure label is an integer
        label_int = int(label)
        # Convert number to digit sequence (e.g., 123 -> [1, 2, 3])
        digits = [int(d) for d in str(label_int)]
        # Pad with zeros to max_digits (e.g., [1, 2, 3] -> [1, 2, 3] for max_digits=3)
        while len(digits) < max_digits:
            digits.append(0)  # Pad with 0
        digits = digits[:max_digits]  # Truncate if longer
        # Clamp digits to [0, vocab_size-1]
        digits = [min(max(d, 0), vocab_size-1) for d in digits]
        # Add START token at beginning and EOS token at end
        sequence = [START_TOKEN] + digits + [EOS_TOKEN]
        sequences.append(sequence)
    seq_tensor = torch.tensor(sequences, dtype=torch.long)
    # Clamp all values to [0, vocab_size-1]
    seq_tensor = torch.clamp(seq_tensor, 0, vocab_size-1)
    return seq_tensor

def sequence_to_number(sequence):
    """Convert digit sequence back to number"""
    numbers = []
    for seq in sequence:
        num = 0
        for digit in seq:
            if digit == 11:  # EOS token
                break
            if digit != 0 and digit != 10:  # Skip padding zeros and START token
                num = num * 10 + digit.item()
        numbers.append(num)
    return numbers

def validate_batch(images, targets, vocab_size):
    assert images.dtype == torch.float32, f"Images dtype is {images.dtype}, expected float32"
    assert torch.isfinite(images).all(), "Non-finite values in images"
    assert targets.dtype == torch.long, f"Targets dtype is {targets.dtype}, expected long"
    assert torch.isfinite(targets.float()).all(), "Non-finite values in targets"
    assert ((targets >= 0) & (targets < vocab_size)).all(), f"Target out of range: min {targets.min().item()}, max {targets.max().item()}"
    
    # Check image statistics
    img_mean = images.mean().item()
    img_std = images.std().item()
    if abs(img_mean) > 1.0 or img_std > 2.0:
        print(f"Warning: Image statistics - mean: {img_mean:.4f}, std: {img_std:.4f}")

def train_seq2seq_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=10, device='cpu'):
    """
    Train the Seq2Seq VisionTransformer model.
    
    Args:
        model: Seq2SeqVisionTransformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        scheduler: Learning rate scheduler
        epochs: Number of training epochs
        batch_size: Batch size
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
            
            # Simple normalization
            images = (images - images.mean()) / (images.std() + 1e-8)
            
            # Convert labels to digit sequences
            target_sequences = create_digit_sequences(labels, max_digits=TRAINING_CONFIG["max_digits"], vocab_size=MODEL_CONFIG["vocab_size"]).to(device)
            
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
                
                # Simple normalization
                images = (images - images.mean()) / (images.std() + 1e-8)
                
                # Convert labels to digit sequences
                target_sequences = create_digit_sequences(labels, max_digits=TRAINING_CONFIG["max_digits"], vocab_size=MODEL_CONFIG["vocab_size"]).to(device)
                
                # Use autoregressive generation for validation
                start_token = 10  # START token
                max_length = TRAINING_CONFIG["max_digits"] + 2  # +2 for START and EOS tokens
                generated_sequences = model.generate(images, start_token=start_token, max_length=max_length)
                
                # Compare generated sequences to target sequences
                for i in range(len(generated_sequences)):
                    pred_num = sequence_to_number(generated_sequences[i].unsqueeze(0))[0]
                    target_num = sequence_to_number(target_sequences[i].unsqueeze(0))[0]
                    if pred_num == target_num:
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
            torch.save(model.state_dict(), MODEL_PATHS["best_seq2seq_model"])
            print(f'New best model saved with accuracy: {best_acc:.2f}%')
        
        print('-' * 50)
    
    return best_acc

def evaluate_seq2seq_model(model, test_loader, device='cpu'):
    """
    Evaluate the trained Seq2Seq VisionTransformer model.
    
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
            
            # Normalize images to prevent numerical instability
            if images.max() > 1.0 or images.min() < -1.0:
                images = torch.clamp(images, -3.0, 3.0)  # Clamp extreme values
                images = (images - images.mean()) / (images.std() + 1e-8)  # Normalize
            
            # Convert labels to sequences
            target_sequences = create_digit_sequences(labels, max_digits=TRAINING_CONFIG["max_digits"], vocab_size=MODEL_CONFIG["vocab_size"]).to(device)
            
            # Use autoregressive generation for evaluation
            start_token = 10  # START token
            max_length = TRAINING_CONFIG["max_digits"] + 2  # +2 for START and EOS tokens
            generated_sequences = model.generate(images, start_token=start_token, max_length=max_length)
            
            # Compare generated sequences to target sequences
            for i in range(len(generated_sequences)):
                pred_num = sequence_to_number(generated_sequences[i].unsqueeze(0))[0]
                target_num = sequence_to_number(target_sequences[i].unsqueeze(0))[0]
                
                if pred_num == target_num:
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
    config = create_seq2seq_config()
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
    best_accuracy = train_seq2seq_model(
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
    torch.save(model.state_dict(), MODEL_PATHS["final_seq2seq_model"])
    print(f"Final model saved to {MODEL_PATHS['final_seq2seq_model']}")
    
    # Evaluate on test set
    print("\nEvaluating final model on test set...")
    evaluate_seq2seq_model(model, test_loader, device=device) 