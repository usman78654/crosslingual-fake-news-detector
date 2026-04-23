"""
Week 3: XLM-RoBERTa Implementation - English Training
Train multilingual transformer on English dataset
Optimized for NVIDIA RTX 5080 GPU
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    XLMRobertaTokenizer, 
    XLMRobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from tqdm import tqdm
import os
import json
from contextlib import nullcontext

def setup_device():
    """Configure and report compute device."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Enable TF32 for faster computation on Ada GPUs.
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print("TensorFloat32: Enabled for faster computation")
    else:
        print("WARNING: CUDA not available, using CPU (training will be slow)")
    print('='*60)

    return device


def get_autocast_context(device, enabled):
    """Create backward-compatible autocast context."""
    if not enabled or device.type != 'cuda':
        return nullcontext()

    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast(device_type='cuda', dtype=torch.float16)

    return torch.cuda.amp.autocast(dtype=torch.float16)


def create_grad_scaler(device, enabled):
    """Create backward-compatible GradScaler."""
    if not enabled or device.type != 'cuda':
        return None

    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda')

    return torch.cuda.amp.GradScaler()

class FakeNewsDataset(Dataset):
    """Dataset class for fake news detection"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, dataloader, optimizer, scheduler, device, scaler=None):
    """Train for one epoch with optional mixed precision"""
    model.train()
    total_loss = 0
    use_mixed_precision = scaler is not None and device.type == 'cuda'
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        if use_mixed_precision:
            # Mixed precision training for faster computation on RTX 5080
            with get_autocast_context(device, enabled=True):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, scaler=None):
    """Evaluate model with optional mixed precision"""
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    use_mixed_precision = scaler is not None and device.type == 'cuda'
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if use_mixed_precision:
                with get_autocast_context(device, enabled=True):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            total_loss += outputs.loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary'
    )
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions,
        'true_labels': true_labels
    }

def main():
    print("\n" + "="*60)
    print("CROSS-LINGUAL FAKE NEWS DETECTION")
    print("Week 3: XLM-RoBERTa Training (English)")
    print("Optimized for NVIDIA RTX 5080")
    print("="*60)

    device = setup_device()
    
    # Create directories
    os.makedirs('results/week3_model_training', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Hyperparameters - Optimized for RTX 5080 (16GB VRAM)
    BATCH_SIZE = 32  # Larger batch size for RTX 5080
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 256
    VALIDATION_SPLIT = 0.1
    EARLY_STOPPING_PATIENCE = 2
    MIN_IMPROVEMENT = 1e-4
    USE_MIXED_PRECISION = True  # Enable for faster training on RTX 5080
    
    print(f"\nHyperparameters (RTX 5080 Optimized):")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Validation split: {VALIDATION_SPLIT}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  Mixed precision: {USE_MIXED_PRECISION}")
    
    # Load data
    print(f"\n[1/6] Loading processed data...")
    train_df = pd.read_csv('data/processed/english_train.csv')
    test_df = pd.read_csv('data/processed/english_test.csv')

    train_split_df, val_df = train_test_split(
        train_df,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=train_df['label']
    )
    train_split_df = train_split_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"  Train samples: {len(train_split_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    # Load tokenizer and model
    print(f"\n[2/6] Loading XLM-RoBERTa model...")
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model = XLMRobertaForSequenceClassification.from_pretrained(
        'xlm-roberta-base',
        num_labels=2
    )
    model.to(device)
    
    # Enable gradient checkpointing to save GPU memory
    if torch.cuda.is_available():
        model.gradient_checkpointing_enable()
    
    print(f"  ✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create datasets
    print(f"\n[3/6] Creating datasets...")
    train_dataset = FakeNewsDataset(
        train_split_df['text'], train_split_df['label'], tokenizer, MAX_LENGTH
    )
    val_dataset = FakeNewsDataset(
        val_df['text'], val_df['label'], tokenizer, MAX_LENGTH
    )
    test_dataset = FakeNewsDataset(
        test_df['text'], test_df['label'], tokenizer, MAX_LENGTH
    )
    
    # Windows uses spawn workers, which re-imports modules and adds overhead.
    num_workers = 0 if os.name == 'nt' else (2 if torch.cuda.is_available() else 0)

    # Use pin_memory for faster GPU data transfer
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Optimizer and scheduler
    print(f"\n[4/6] Setting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Mixed precision scaler
    scaler = create_grad_scaler(device, USE_MIXED_PRECISION)
    if scaler:
        print("  ✓ Mixed precision training enabled")
    
    # Training loop
    print(f"\n[5/6] Training model...")
    print("="*60)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 60)
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        print(f"Training loss: {train_loss:.4f}")
        
        val_metrics = evaluate(model, val_loader, device, scaler)
        print(f"Validation loss: {val_metrics['loss']:.4f}")
        print(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Validation F1: {val_metrics['f1']:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])

        # Keep the best model by validation F1 and stop when it plateaus.
        current_val_f1 = val_metrics['f1']
        if current_val_f1 > best_val_f1 + MIN_IMPROVEMENT:
            best_val_f1 = current_val_f1
            epochs_without_improvement = 0
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
            print(
                f"Validation F1 not improving "
                f"(patience {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE})."
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping: validation F1 plateaued. "
                f"Best validation F1: {best_val_f1:.4f}"
            )
            break
        
        # Clear GPU cache to prevent memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\nLoaded best checkpoint based on validation F1.")
    
    # Save model
    print(f"\n[6/6] Saving model...")
    model_path = 'models/xlm_roberta_english'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"  ✓ Model saved to {model_path}")
    
    # Save history
    with open('results/week3_model_training/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Final English Model Performance (Held-Out Test Set)")
    print('='*60)
    
    final_metrics = evaluate(model, test_loader, device, scaler)
    
    print(f"\nMetrics:")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall: {final_metrics['recall']:.4f}")
    print(f"  F1-Score: {final_metrics['f1']:.4f}")
    
    # Save metrics
    metrics_dict = {
        'accuracy': float(final_metrics['accuracy']),
        'precision': float(final_metrics['precision']),
        'recall': float(final_metrics['recall']),
        'f1': float(final_metrics['f1']),
        'best_validation_f1': float(best_val_f1),
        'selection_metric': 'validation_f1',
        'data_split': {
            'train_samples': int(len(train_split_df)),
            'validation_samples': int(len(val_df)),
            'test_samples': int(len(test_df))
        }
    }
    
    with open('results/week3_model_training/english_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    # Confusion Matrix and Visualizations
    cm = confusion_matrix(final_metrics['true_labels'], final_metrics['predictions'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training curves - Fixed to use actual history length
    epochs_range = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs_range, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Fake', 'True'],
                yticklabels=['Fake', 'True'])
    axes[1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('results/week3_model_training/english_model_results.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualizations saved")
    
    print("\n" + "="*60)
    print("✅ Week 3 - English Model Training Complete!")
    print("="*60)
    print("\nDeliverables:")
    print("  ✔ XLM-RoBERTa trained on English data (RTX 5080 optimized)")
    print(f"  ✔ Model accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  ✔ Model F1-score: {final_metrics['f1']:.4f}")
    print("  ✔ Model saved for cross-lingual testing")
    print("\nNext: Week 4 - Cross-Lingual Evaluation")

if __name__ == "__main__":
    main()
