import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from tqdm import tqdm
from vit_model import VisionTransformer
from vit_dataset import MultiDigitMNISTDataset, create_sample_visualization
from config import MODEL_CONFIG, DATASET_CONFIG, MODEL_PATHS

def load_model(model_path: str, model_config: dict):
    """
    Load a trained ViT model.
    
    Args:
        model_path: Path to the saved model parameters
        model_config: Model configuration dictionary
        
    Returns:
        Loaded VisionTransformer model
    """
    # Initialize model
    model = VisionTransformer(**model_config)
    
    # Load parameters
    if os.path.exists(model_path):
        params = torch.load(model_path)
        model.load_state_dict(params)
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Using untrained model.")
    
    return model

def evaluate_model(model, dataset, split='val', batch_size=64, device='cpu'):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained VisionTransformer model
        dataset: MultiDigitMNISTDataset instance
        split: 'train' or 'val'
        batch_size: Batch size for evaluation
        device: Device to evaluate on
    """
    model.eval()
    
    # Get all data
    images, labels = dataset.get_all_data(split)
    images = torch.tensor(np.array(images), dtype=torch.float32)
    labels = torch.tensor(np.array(labels), dtype=torch.long)
    
    # Add channel dimension (batch_size, channels, height, width)
    images = images.unsqueeze(1)
    
    # Convert labels to digit sequences
    def create_digit_sequences(labels, max_digits=3):
        sequences = []
        for label in labels:
            # Ensure label is an integer
            label_int = int(label)
            # Convert number to digit sequence (e.g., 123 -> [1, 2, 3])
            digits = [int(d) for d in str(label_int)]
            # Pad with zeros to max_digits (e.g., [1, 2, 3] -> [1, 2, 3] for max_digits=3)
            while len(digits) < max_digits:
                digits.append(0)  # Pad with 0
            sequences.append(digits[:max_digits])  # Truncate if longer
        return torch.tensor(sequences, dtype=torch.long)
    
    sequences = create_digit_sequences(labels, max_digits=MODEL_CONFIG["max_digits"])
    
    # Evaluate in batches
    n_batches = len(images) // batch_size
    correct = 0
    total = 0
    
    # Progress bar for evaluation
    eval_pbar = tqdm(range(n_batches), desc=f'Evaluating {split}', leave=False)
    
    with torch.no_grad():
        for batch_idx in eval_pbar:
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_images = images[start_idx:end_idx].to(device)
            batch_sequences = sequences[start_idx:end_idx].to(device)
            
            # Forward pass
            outputs = model(batch_images)
            predicted_digits = torch.argmax(outputs, dim=-1)  # (batch_size, max_digits)
            
            # Convert sequences back to numbers for comparison
            for i in range(len(predicted_digits)):
                # Convert predicted digits to number
                pred_num = 0
                for j, digit in enumerate(predicted_digits[i]):
                    if digit != 0:  # Skip padding zeros
                        pred_num = pred_num * 10 + digit.item()
                
                # Convert target sequence to number
                target_num = 0
                for j, digit in enumerate(batch_sequences[i]):
                    if digit != 0:  # Skip padding zeros
                        target_num = target_num * 10 + digit.item()
                
                if pred_num == target_num:
                    correct += 1
                total += 1
            
            # Update progress bar
            current_acc = 100 * correct / total
            eval_pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
    
    accuracy = 100 * correct / total
    print(f'{split.capitalize()} Accuracy: {accuracy:.2f}%')
    return accuracy

def print_evaluation_results(results, split_name):
    """Print formatted evaluation results."""
    print(f"\n{'='*50}")
    print(f"{split_name.upper()} EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {results['accuracy']:.2f}%")
    print(f"Total Samples: {results['total_samples']:,}")
    print(f"Correct Predictions: {results['correct_predictions']:,}")
    
    print(f"\nPer-Digit Accuracy:")
    for digit_type, acc in results['digit_accuracies'].items():
        print(f"  {digit_type}: {acc:.2f}%")
    
    print(f"{'='*50}")

def analyze_errors(model, dataset, split='val', n_errors=10):
    """
    Analyze prediction errors to understand model weaknesses.
    
    Args:
        model: Trained VisionTransformer model
        dataset: MultiDigitMNISTDataset instance
        split: 'train' or 'val'
        n_errors: Number of error cases to analyze
    """
    model.eval()
    
    # Get all data
    images, labels = dataset.get_all_data(split)
    
    # Get predictions
    logits = model(images)
    predictions = torch.argmax(logits, dim=-1)
    
    # Find errors
    errors = predictions != labels
    error_indices = torch.where(errors)[0]
    
    if len(error_indices) == 0:
        print("No errors found!")
        return
    
    print(f"\nAnalyzing {min(n_errors, len(error_indices))} error cases:")
    print("-" * 60)
    
    for i in range(min(n_errors, len(error_indices))):
        idx = error_indices[i].item()
        true_label = labels[idx].item()
        pred_label = predictions[idx].item()
        confidence = torch.softmax(logits[idx], dim=0)[pred_label].item()
        
        print(f"Sample {i+1}:")
        print(f"  True: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.3f}")
        print(f"  True digits: {list(str(true_label))}")
        print(f"  Pred digits: {list(str(pred_label))}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Evaluate ViT model on multi-digit MNIST')
    parser.add_argument('--model_path', type=str, default='model_data/best_multi_digit_vit.npz',
                       help='Path to trained model parameters')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing MNIST data')
    parser.add_argument('--image_size', type=int, default=84,
                       help='Image size for the model')
    parser.add_argument('--max_digits', type=int, default=3,
                       help='Maximum number of digits per image')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--create_visualization', action='store_true',
                       help='Create sample visualization')
    parser.add_argument('--analyze_errors', action='store_true',
                       help='Analyze prediction errors')
    
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = MultiDigitMNISTDataset(
        data_dir=DATASET_CONFIG["data_dir"],
        image_size=DATASET_CONFIG["image_size"],
        max_digits=DATASET_CONFIG["max_digits"],
        train_split=DATASET_CONFIG["train_split"]
    )
    
    print(f"Training samples: {len(dataset.train_images)}")
    print(f"Validation samples: {len(dataset.val_images)}")
    print(f"Label range: {dataset.train_labels.min()} - {dataset.train_labels.max()}")

    # Create model with configuration from config file
    model = VisionTransformer(**MODEL_CONFIG)
    
    # Load trained model
    model_path = MODEL_PATHS["best_model"]
    try:
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Evaluate model
    print("\nEvaluating model...")
    train_accuracy = evaluate_model(model, dataset, split='train', batch_size=64, device=device)
    val_accuracy = evaluate_model(model, dataset, split='val', batch_size=64, device=device)
    
    print(f"\nFinal Results:")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

if __name__ == "__main__":
    main() 