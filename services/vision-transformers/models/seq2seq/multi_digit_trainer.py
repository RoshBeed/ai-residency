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
import math

# Ensure project root is in sys.path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from .seq2seq import Seq2SeqVisionTransformer
from datasets.multi_digit_dataset import MultiDigitMNISTDataset
from config import MODEL_CONFIG, TRAINING_CONFIG, DATASET_CONFIG, MODEL_PATHS

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing"""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_last_lr(self):
        return [self.optimizer.param_groups[0]['lr']]

def label_smoothing_loss(predictions, targets, smoothing=0.1, num_classes=None):
    """
    Compute label smoothing loss
    
    Args:
        predictions: (batch_size, num_classes)
        targets: (batch_size,)
        smoothing: Label smoothing factor
        num_classes: Number of classes
    """
    if num_classes is None:
        num_classes = predictions.size(-1)
    
    # Create one-hot encoded targets
    one_hot = torch.zeros_like(predictions).scatter_(1, targets.unsqueeze(1), 1)
    
    # Apply label smoothing
    smooth_targets = one_hot * (1 - smoothing) + smoothing / num_classes
    
    # Compute cross-entropy loss
    log_probs = torch.log_softmax(predictions, dim=-1)
    loss = -torch.sum(smooth_targets * log_probs, dim=-1)
    
    return loss.mean()

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
    
    # Label smoothing for better generalization
    label_smoothing = TRAINING_CONFIG.get("label_smoothing", 0.0)
    
    # Gradient accumulation for memory efficiency
    gradient_accumulation_steps = TRAINING_CONFIG.get("gradient_accumulation_steps", 1)
    
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
            
            # Remove redundant normalization - dataset already normalizes
            # images = (images - images.mean()) / (images.std() + 1e-8)
            
            # Data augmentation during training
            if model.training:
                # Random rotation (small angles)
                if torch.rand(1, device=images.device) < 0.3:
                    angle = torch.rand(1, device=images.device) * 10 - 5  # -5 to +5 degrees
                    images = torch.rot90(images, k=int(angle.item() // 90), dims=[2, 3])
                
                # Random brightness adjustment
                if torch.rand(1, device=images.device) < 0.3:
                    brightness_factor = 0.8 + torch.rand(1, device=images.device) * 0.4  # 0.8 to 1.2
                    images = images * brightness_factor
                    images = torch.clamp(images, -1, 1)
            
            # Convert labels to digit sequences
            target_sequences = labels.to(device)  # Labels are already digit sequences
            
            # Forward pass
            outputs = model(images, target_sequences)
            
            # Calculate loss for each position with label smoothing
            if label_smoothing > 0:
                # Apply label smoothing using custom function
                loss = 0
                seq_len = outputs.size(1)
                for pos in range(seq_len):
                    pos_output = outputs[:, pos, :]
                    pos_target = target_sequences[:, pos]
                    pos_loss = label_smoothing_loss(pos_output, pos_target, label_smoothing, MODEL_CONFIG["vocab_size"])
                    loss += pos_loss
                loss = loss / seq_len
            else:
                # Use efficient cross-entropy with reduction='none'
                # Reshape outputs: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = target_sequences.view(-1)
                
                # Compute loss for all positions at once
                loss = torch.nn.functional.cross_entropy(outputs_flat, targets_flat, reduction='mean')
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TRAINING_CONFIG["gradient_clip_norm"])
                
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * gradient_accumulation_steps
            
            # Update progress bar
            train_pbar.set_postfix({'Loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})
        
        # Ensure gradients are applied at the end of epoch if needed
        if len(train_loader) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TRAINING_CONFIG["gradient_clip_norm"])
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation phase
        model.eval()
        correct_autoregressive = 0
        correct_teacher_forcing = 0
        total = 0
        
        # Progress bar for validation batches
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_pbar):
                images = images.to(device)
                labels = labels.to(device)
                
                # Remove redundant normalization - dataset already normalizes
                # images = (images - images.mean()) / (images.std() + 1e-8)
                
                # Convert labels to digit sequences
                target_sequences = labels.to(device)  # Labels are already digit sequences
                
                # Teacher forcing validation (for comparison)
                outputs_teacher = model(images, target_sequences)
                outputs_flat = outputs_teacher.view(-1, outputs_teacher.size(-1))
                targets_flat = target_sequences.view(-1)
                _, predicted_teacher = torch.max(outputs_flat, 1)
                
                # Autoregressive generation validation
                start_token = MODEL_CONFIG["start_token"]  # START token
                max_length = TRAINING_CONFIG["max_digits"] + 2  # +2 for START and EOS tokens
                generated_sequences = model.generate(images, start_token=start_token, max_length=max_length)
                
                # Compare generated sequences to target sequences
                for i in range(len(generated_sequences)):
                    pred_num = sequence_to_number(generated_sequences[i].unsqueeze(0))[0]
                    target_num = sequence_to_number(target_sequences[i].unsqueeze(0))[0]
                    if pred_num == target_num:
                        correct_autoregressive += 1
                    total += 1
                
                # Teacher forcing accuracy (per-token)
                correct_tokens = (predicted_teacher == targets_flat).sum().item()
                correct_teacher_forcing += correct_tokens
                
                # Update validation progress bar
                current_acc_auto = 100 * correct_autoregressive / total
                current_acc_teacher = 100 * correct_teacher_forcing / (total * (TRAINING_CONFIG["max_digits"] + 2))
                val_pbar.set_postfix({
                    'Auto Acc': f'{current_acc_auto:.1f}%',
                    'Teacher Acc': f'{current_acc_teacher:.1f}%'
                })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc_auto = 100 * correct_autoregressive / total
        epoch_acc_teacher = 100 * correct_teacher_forcing / (total * (TRAINING_CONFIG["max_digits"] + 2))
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s')
        print(f'Training Loss: {epoch_loss:.4f}')
        print(f'Validation Accuracy (Autoregressive): {epoch_acc_auto:.2f}%')
        print(f'Validation Accuracy (Teacher Forcing): {epoch_acc_teacher:.2f}%')
        
        # Step the scheduler based on autoregressive validation accuracy
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f'Learning Rate: {current_lr:.6f}')
        
        # Save best model based on autoregressive accuracy
        if epoch_acc_auto > best_acc:
            best_acc = epoch_acc_auto
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
            
            # Remove redundant normalization - dataset already normalizes
            # if images.max() > 1.0 or images.min() < -1.0:
            #     images = torch.clamp(images, -3.0, 3.0)  # Clamp extreme values
            #     images = (images - images.mean()) / (images.std() + 1e-8)  # Normalize
            
            # Convert labels to sequences
            target_sequences = labels.to(device)  # Labels are already digit sequences
            
            # Use autoregressive generation for evaluation
            start_token = MODEL_CONFIG["start_token"]  # START token
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
        samples_per_class=100,  # Increased from 10 to 100 for paper ratios
        batch_size=TRAINING_CONFIG["batch_size"],
        num_workers=4,
        download=True,
        return_digit_sequence=True,
        add_start_eos=True
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
    
    # Calculate and print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Estimate memory usage for a batch
    sample_batch_size = TRAINING_CONFIG["batch_size"]
    sample_seq_len = TRAINING_CONFIG["max_digits"] + 2
    estimated_memory = (total_params * 4 +  # Model parameters
                       sample_batch_size * MODEL_CONFIG["img_size"]**2 * 4 +  # Input images
                       sample_batch_size * sample_seq_len * MODEL_CONFIG["d_model"] * 4 * 3) / 1024 / 1024  # Activations
    print(f"  Estimated memory per batch: {estimated_memory:.2f} MB")
    print()
    
    # Create optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"]
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Calculate total training steps for scheduler
    total_steps = len(train_loader) * TRAINING_CONFIG["epochs"]
    
    # Create scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        total_steps=total_steps
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