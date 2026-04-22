"""
Week 3: XLM-RoBERTa Implementation - English Training
Train multilingual transformer on English dataset
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
import seaborn as sns
from tqdm import tqdm
import os
import json

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print('='*60)

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

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
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
    print("="*60)
    
    # Create directories
    os.makedirs('results/week3_model_training', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 256
    
    print(f"\nHyperparameters:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max length: {MAX_LENGTH}")
    
    # Load data
    print(f"\n[1/6] Loading processed data...")
    train_df = pd.read_csv('data/processed/english_train.csv')
    test_df = pd.read_csv('data/processed/english_test.csv')
    
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    # Load tokenizer and model
    print(f"\n[2/6] Loading XLM-RoBERTa model...")
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model = XLMRobertaForSequenceClassification.from_pretrained(
        'xlm-roberta-base',
        num_labels=2
    )
    model.to(device)
    
    print(f"  ✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create datasets
    print(f"\n[3/6] Creating datasets...")
    train_dataset = FakeNewsDataset(
        train_df['text'], train_df['label'], tokenizer, MAX_LENGTH
    )
    test_dataset = FakeNewsDataset(
        test_df['text'], test_df['label'], tokenizer, MAX_LENGTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    print(f"  Train batches: {len(train_loader)}")
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
    
    # Training loop
    print(f"\n[5/6] Training model...")
    print("="*60)
    
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': [],
        'test_f1': []
    }
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 60)
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Training loss: {train_loss:.4f}")
        
        test_metrics = evaluate(model, test_loader, device)
        print(f"Test loss: {test_metrics['loss']:.4f}")
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_metrics['loss'])
        history['test_accuracy'].append(test_metrics['accuracy'])
        history['test_f1'].append(test_metrics['f1'])
    
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
    print("Final English Model Performance")
    print('='*60)
    
    final_metrics = evaluate(model, test_loader, device)
    
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
        'f1': float(final_metrics['f1'])
    }
    
    with open('results/week3_model_training/english_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    # Confusion Matrix
    cm = confusion_matrix(final_metrics['true_labels'], final_metrics['predictions'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training curves
    epochs_range = range(1, EPOCHS + 1)
    axes[0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs_range, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    axes[0].set_title('Training and Test Loss', fontsize=14, fontweight='bold')
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
    print("  ✔ XLM-RoBERTa trained on English data")
    print(f"  ✔ Model accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  ✔ Model F1-score: {final_metrics['f1']:.4f}")
    print("  ✔ Model saved for cross-lingual testing")
    print("\nNext: Week 4 - Cross-Lingual Evaluation")

if __name__ == "__main__":
    main()
