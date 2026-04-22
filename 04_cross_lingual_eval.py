"""
Week 4: Cross-Lingual Transfer Learning Evaluation
Test English-trained model on Urdu dataset (zero-shot)
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
import seaborn as sns
from tqdm import tqdm
import os
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def evaluate(model, dataloader, device, dataset_name):
    """Evaluate model on given dataset"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Evaluating {dataset_name}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary'
    )
    
    return {
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
    print("Week 4: Cross-Lingual Transfer Evaluation")
    print("="*60)
    
    # Create directories
    os.makedirs('results/week4_cross_lingual', exist_ok=True)
    
    # Load model trained on English
    print(f"\n[1/5] Loading English-trained model...")
    model_path = 'models/xlm_roberta_english'
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
    model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print(f"  ✓ Model loaded from {model_path}")
    
    # Load test datasets
    print(f"\n[2/5] Loading test datasets...")
    english_test = pd.read_csv('data/processed/english_test.csv')
    urdu_test = pd.read_csv('data/processed/urdu_test.csv')
    
    print(f"  English test: {len(english_test)} samples")
    print(f"  Urdu test: {len(urdu_test)} samples")
    
    # Create dataloaders
    print(f"\n[3/5] Creating dataloaders...")
    BATCH_SIZE = 16
    MAX_LENGTH = 256
    
    english_dataset = FakeNewsDataset(
        english_test['text'], english_test['label'], tokenizer, MAX_LENGTH
    )
    urdu_dataset = FakeNewsDataset(
        urdu_test['text'], urdu_test['label'], tokenizer, MAX_LENGTH
    )
    
    english_loader = DataLoader(english_dataset, batch_size=BATCH_SIZE)
    urdu_loader = DataLoader(urdu_dataset, batch_size=BATCH_SIZE)
    
    # Evaluate on both languages
    print(f"\n[4/5] Evaluating on both languages...")
    print("="*60)
    
    # English (same language)
    print(f"\n📊 Testing on English (same language as training)...")
    english_results = evaluate(model, english_loader, device, 'English')
    
    print(f"\nEnglish Test Results:")
    print(f"  Accuracy: {english_results['accuracy']:.4f}")
    print(f"  Precision: {english_results['precision']:.4f}")
    print(f"  Recall: {english_results['recall']:.4f}")
    print(f"  F1-Score: {english_results['f1']:.4f}")
    
    # Urdu (zero-shot cross-lingual)
    print(f"\n📊 Testing on Urdu (zero-shot cross-lingual)...")
    urdu_results = evaluate(model, urdu_loader, device, 'Urdu')
    
    print(f"\nUrdu Test Results (Zero-Shot):")
    print(f"  Accuracy: {urdu_results['accuracy']:.4f}")
    print(f"  Precision: {urdu_results['precision']:.4f}")
    print(f"  Recall: {urdu_results['recall']:.4f}")
    print(f"  F1-Score: {urdu_results['f1']:.4f}")
    
    # Performance degradation analysis
    print(f"\n{'='*60}")
    print("Cross-Lingual Transfer Analysis")
    print('='*60)
    
    accuracy_drop = english_results['accuracy'] - urdu_results['accuracy']
    f1_drop = english_results['f1'] - urdu_results['f1']
    
    print(f"\nPerformance Degradation:")
    print(f"  Accuracy drop: {accuracy_drop:.4f} ({accuracy_drop/english_results['accuracy']*100:.2f}%)")
    print(f"  F1-score drop: {f1_drop:.4f} ({f1_drop/english_results['f1']*100:.2f}%)")
    
    # Save results
    results_dict = {
        'english': {
            'accuracy': float(english_results['accuracy']),
            'precision': float(english_results['precision']),
            'recall': float(english_results['recall']),
            'f1': float(english_results['f1'])
        },
        'urdu_zero_shot': {
            'accuracy': float(urdu_results['accuracy']),
            'precision': float(urdu_results['precision']),
            'recall': float(urdu_results['recall']),
            'f1': float(urdu_results['f1'])
        },
        'performance_drop': {
            'accuracy': float(accuracy_drop),
            'f1': float(f1_drop)
        }
    }
    
    with open('results/week4_cross_lingual/cross_lingual_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    # Visualization
    print(f"\n[5/5] Creating visualizations...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Performance comparison
    ax1 = fig.add_subplot(gs[0, :2])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    english_vals = [english_results['accuracy'], english_results['precision'], 
                   english_results['recall'], english_results['f1']]
    urdu_vals = [urdu_results['accuracy'], urdu_results['precision'],
                urdu_results['recall'], urdu_results['f1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, english_vals, width, label='English (Same Language)', color='#3498db')
    ax1.bar(x + width/2, urdu_vals, width, label='Urdu (Zero-Shot)', color='#e74c3c')
    ax1.set_ylabel('Score')
    ax1.set_title('Cross-Lingual Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Add value labels
    for i, (ev, uv) in enumerate(zip(english_vals, urdu_vals)):
        ax1.text(i - width/2, ev + 0.02, f'{ev:.3f}', ha='center', fontsize=9)
        ax1.text(i + width/2, uv + 0.02, f'{uv:.3f}', ha='center', fontsize=9)
    
    # 2. Performance drop
    ax2 = fig.add_subplot(gs[0, 2])
    drops = [accuracy_drop, f1_drop]
    ax2.bar(['Accuracy\nDrop', 'F1\nDrop'], drops, color='#e67e22')
    ax2.set_title('Performance Degradation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score Drop')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(drops):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # 3. English confusion matrix
    ax3 = fig.add_subplot(gs[1, 0])
    cm_english = confusion_matrix(english_results['true_labels'], english_results['predictions'])
    sns.heatmap(cm_english, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
    ax3.set_title('English Confusion Matrix', fontsize=12, fontweight='bold')
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    
    # 4. Urdu confusion matrix
    ax4 = fig.add_subplot(gs[1, 1])
    cm_urdu = confusion_matrix(urdu_results['true_labels'], urdu_results['predictions'])
    sns.heatmap(cm_urdu, annot=True, fmt='d', cmap='Reds', ax=ax4,
                xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
    ax4.set_title('Urdu Confusion Matrix (Zero-Shot)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('True Label')
    ax4.set_xlabel('Predicted Label')
    
    # 5. Classification reports
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    report_text = "Classification Reports\n\n"
    report_text += "ENGLISH:\n"
    report_text += classification_report(
        english_results['true_labels'], 
        english_results['predictions'],
        target_names=['Fake', 'True'],
        digits=3
    )
    report_text += "\n\nURDU (Zero-Shot):\n"
    report_text += classification_report(
        urdu_results['true_labels'],
        urdu_results['predictions'],
        target_names=['Fake', 'True'],
        digits=3
    )
    
    ax5.text(0.1, 0.5, report_text, fontsize=8, family='monospace',
            verticalalignment='center', transform=ax5.transAxes)
    
    plt.savefig('results/week4_cross_lingual/cross_lingual_evaluation.png',
                dpi=300, bbox_inches='tight')
    print(f"  ✓ Visualizations saved")
    
    # Summary
    print("\n" + "="*60)
    print("✅ Week 4 - Cross-Lingual Evaluation Complete!")
    print("="*60)
    print("\nKey Findings:")
    print(f"  ✔ English model achieves {english_results['accuracy']:.4f} accuracy")
    print(f"  ✔ Zero-shot Urdu accuracy: {urdu_results['accuracy']:.4f}")
    print(f"  ✔ Performance drop: {accuracy_drop:.4f} ({accuracy_drop/english_results['accuracy']*100:.2f}%)")
    print(f"  ✔ Cross-lingual transfer capability demonstrated")
    print("\nNext: Week 5 - Joint Training & Fine-Tuning")

if __name__ == "__main__":
    main()
