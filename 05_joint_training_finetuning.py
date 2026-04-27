"""
Week 5: Joint Training & Fine-Tuning
1. Train on combined multilingual dataset
2. Fine-tune English model on Urdu
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FakeNewsDataset(Dataset):
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
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
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
        true_labels, predictions, average='binary', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions,
        'true_labels': true_labels
    }

def train_model(model, train_loader, eval_loader_dict, optimizer, scheduler, 
                epochs, device, model_name, target_accuracy=0.80,
                early_stopping_patience=2, min_improvement=1e-4,
                primary_language='urdu'):
    """Train model and track performance with target-aware early stopping."""
    
    history = {
        'train_loss': [],
        'english_val_acc': [],
        'urdu_val_acc': [],
        'english_val_f1': [],
        'urdu_val_f1': []
    }

    best_primary_accuracy = -1.0
    epochs_without_improvement = 0
    best_model_state = None
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 60)
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Training loss: {train_loss:.4f}")
        
        epoch_metrics = {}

        # Evaluate on both validation sets
        for lang, loader in eval_loader_dict.items():
            metrics = evaluate(model, loader, device)
            print(f"{lang.capitalize()} validation - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
            epoch_metrics[lang] = metrics
            
            if lang == 'english':
                history['english_val_acc'].append(metrics['accuracy'])
                history['english_val_f1'].append(metrics['f1'])
            else:
                history['urdu_val_acc'].append(metrics['accuracy'])
                history['urdu_val_f1'].append(metrics['f1'])
        
        history['train_loss'].append(train_loss)

        # Track early stopping using primary validation language (Urdu by default).
        primary_metrics = epoch_metrics.get(primary_language)
        if primary_metrics is None:
            # Fallback to the first available language if primary is missing.
            primary_lang, primary_metrics = next(iter(epoch_metrics.items()))
        else:
            primary_lang = primary_language

        primary_accuracy = primary_metrics['accuracy']

        if primary_accuracy > best_primary_accuracy + min_improvement:
            best_primary_accuracy = primary_accuracy
            epochs_without_improvement = 0
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            if best_primary_accuracy >= target_accuracy:
                epochs_without_improvement += 1
                print(
                    f"{primary_lang.capitalize()} validation accuracy plateaued above target "
                    f"(best={best_primary_accuracy:.4f}, target={target_accuracy:.4f}, "
                    f"patience {epochs_without_improvement}/{early_stopping_patience})."
                )

                if epochs_without_improvement >= early_stopping_patience:
                    print(
                        f"Early stopping {model_name}: {primary_lang} validation accuracy "
                        f"plateaued above {target_accuracy:.2f}."
                    )
                    break
            else:
                print(
                    f"{primary_lang.capitalize()} validation accuracy not improved, "
                    f"but target {target_accuracy:.4f} not reached yet; continuing."
                )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(
            f"Loaded best {primary_language} validation checkpoint "
            f"(accuracy={best_primary_accuracy:.4f})."
        )
    
    return history

def main():
    print("\n" + "="*60)
    print("CROSS-LINGUAL FAKE NEWS DETECTION")
    print("Week 5: Joint Training & Fine-Tuning")
    print("="*60)
    
    # Create directories
    os.makedirs('results/week5_joint_training', exist_ok=True)
    os.makedirs('models/multilingual', exist_ok=True)
    os.makedirs('models/finetuned_urdu', exist_ok=True)
    
    BATCH_SIZE = 16
    MAX_LENGTH = 256
    LEARNING_RATE = 2e-5
    VALIDATION_SPLIT = 0.1
    TARGET_ACCURACY = 0.80
    EARLY_STOPPING_PATIENCE = 2
    MIN_IMPROVEMENT = 1e-4
    
    # ========== Experiment 1: Joint Multilingual Training ==========
    print("\n" + "="*60)
    print("EXPERIMENT 1: Joint Multilingual Training")
    print("="*60)
    
    print(f"\n[1/7] Loading datasets for joint training...")
    english_train = pd.read_csv('data/processed/english_train.csv')
    english_test = pd.read_csv('data/processed/english_test.csv')
    urdu_train = pd.read_csv('data/processed/urdu_train.csv')
    urdu_test = pd.read_csv('data/processed/urdu_test.csv')
    
    # Sample English to balance with Urdu
    if len(english_train) > len(urdu_train) * 3:
        english_train = english_train.sample(n=len(urdu_train) * 3, random_state=42)
        print(f"  Sampled English to {len(english_train)} samples for balance")

    english_train_split, english_val = train_test_split(
        english_train,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=english_train['label']
    )
    urdu_train_split, urdu_val = train_test_split(
        urdu_train,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=urdu_train['label']
    )

    english_train_split = english_train_split.reset_index(drop=True)
    english_val = english_val.reset_index(drop=True)
    urdu_train_split = urdu_train_split.reset_index(drop=True)
    urdu_val = urdu_val.reset_index(drop=True)
    
    # Combine datasets
    combined_train = pd.concat([english_train_split, urdu_train_split], ignore_index=True)
    combined_train = combined_train.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"  Combined train: {len(combined_train)} samples")
    print(f"    English train: {len(english_train_split)}, Urdu train: {len(urdu_train_split)}")
    print(f"  Validation - English: {len(english_val)}, Urdu: {len(urdu_val)}")
    print(f"  English test: {len(english_test)} samples")
    print(f"  Urdu test: {len(urdu_test)} samples")
    
    print(f"\n[2/7] Loading model for joint training...")
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model_joint = XLMRobertaForSequenceClassification.from_pretrained(
        'xlm-roberta-base',
        num_labels=2
    )
    model_joint.to(device)
    
    print(f"\n[3/7] Creating datasets...")
    train_dataset = FakeNewsDataset(
        combined_train['text'], combined_train['label'], tokenizer, MAX_LENGTH
    )
    english_val_dataset = FakeNewsDataset(
        english_val['text'], english_val['label'], tokenizer, MAX_LENGTH
    )
    urdu_val_dataset = FakeNewsDataset(
        urdu_val['text'], urdu_val['label'], tokenizer, MAX_LENGTH
    )
    english_test_dataset = FakeNewsDataset(
        english_test['text'], english_test['label'], tokenizer, MAX_LENGTH
    )
    urdu_test_dataset = FakeNewsDataset(
        urdu_test['text'], urdu_test['label'], tokenizer, MAX_LENGTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    english_val_loader = DataLoader(english_val_dataset, batch_size=BATCH_SIZE)
    urdu_val_loader = DataLoader(urdu_val_dataset, batch_size=BATCH_SIZE)
    english_test_loader = DataLoader(english_test_dataset, batch_size=BATCH_SIZE)
    urdu_test_loader = DataLoader(urdu_test_dataset, batch_size=BATCH_SIZE)
    
    print(f"\n[4/7] Training joint multilingual model...")
    EPOCHS_JOINT = 8
    optimizer = AdamW(model_joint.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS_JOINT
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    history_joint = train_model(
        model_joint, train_loader,
        {'english': english_val_loader, 'urdu': urdu_val_loader},
        optimizer, scheduler, EPOCHS_JOINT, device, 'joint',
        target_accuracy=TARGET_ACCURACY,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        min_improvement=MIN_IMPROVEMENT,
        primary_language='urdu'
    )
    
    # Save joint model
    model_joint.save_pretrained('models/multilingual/xlm_roberta_joint')
    tokenizer.save_pretrained('models/multilingual/xlm_roberta_joint')
    print(f"\n✓ Joint model saved")
    
    # Evaluate joint model
    print(f"\n{'='*60}")
    print("Joint Model Final Results")
    print('='*60)
    
    joint_english_metrics = evaluate(model_joint, english_test_loader, device)
    joint_urdu_metrics = evaluate(model_joint, urdu_test_loader, device)
    
    print(f"\nEnglish: Acc={joint_english_metrics['accuracy']:.4f}, F1={joint_english_metrics['f1']:.4f}")
    print(f"Urdu: Acc={joint_urdu_metrics['accuracy']:.4f}, F1={joint_urdu_metrics['f1']:.4f}")
    
    # ========== Experiment 2: Fine-Tuning on Urdu ==========
    print("\n" + "="*60)
    print("EXPERIMENT 2: Fine-Tuning English Model on Urdu")
    print("="*60)
    
    print(f"\n[5/7] Loading English-trained model...")
    model_finetuned = XLMRobertaForSequenceClassification.from_pretrained(
        'models/xlm_roberta_english'
    )
    model_finetuned.to(device)
    
    print(f"\n[6/7] Fine-tuning on Urdu dataset...")
    urdu_train_dataset = FakeNewsDataset(
        urdu_train_split['text'], urdu_train_split['label'], tokenizer, MAX_LENGTH
    )
    urdu_train_loader = DataLoader(urdu_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    EPOCHS_FINETUNE = 8
    optimizer_ft = AdamW(model_finetuned.parameters(), lr=1e-5)  # Lower LR for fine-tuning
    total_steps_ft = len(urdu_train_loader) * EPOCHS_FINETUNE
    scheduler_ft = get_linear_schedule_with_warmup(
        optimizer_ft, num_warmup_steps=0, num_training_steps=total_steps_ft
    )
    
    history_finetune = train_model(
        model_finetuned, urdu_train_loader,
        {'english': english_val_loader, 'urdu': urdu_val_loader},
        optimizer_ft, scheduler_ft, EPOCHS_FINETUNE, device, 'finetuned',
        target_accuracy=TARGET_ACCURACY,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        min_improvement=MIN_IMPROVEMENT,
        primary_language='urdu'
    )
    
    # Save fine-tuned model
    model_finetuned.save_pretrained('models/finetuned_urdu/xlm_roberta_finetuned')
    tokenizer.save_pretrained('models/finetuned_urdu/xlm_roberta_finetuned')
    print(f"\n✓ Fine-tuned model saved")
    
    # Evaluate fine-tuned model
    print(f"\n{'='*60}")
    print("Fine-Tuned Model Final Results")
    print('='*60)
    
    ft_english_metrics = evaluate(model_finetuned, english_test_loader, device)
    ft_urdu_metrics = evaluate(model_finetuned, urdu_test_loader, device)
    
    print(f"\nEnglish: Acc={ft_english_metrics['accuracy']:.4f}, F1={ft_english_metrics['f1']:.4f}")
    print(f"Urdu: Acc={ft_urdu_metrics['accuracy']:.4f}, F1={ft_urdu_metrics['f1']:.4f}")
    
    # ========== Comparison ==========
    print(f"\n[7/7] Creating comparative analysis...")
    
    # Load previous results
    with open('results/week4_cross_lingual/cross_lingual_results.json', 'r') as f:
        week4_results = json.load(f)
    
    comparison = {
        'english_only': week4_results['urdu_zero_shot'],
        'joint_multilingual': {
            'accuracy': float(joint_urdu_metrics['accuracy']),
            'precision': float(joint_urdu_metrics['precision']),
            'recall': float(joint_urdu_metrics['recall']),
            'f1': float(joint_urdu_metrics['f1'])
        },
        'finetuned': {
            'accuracy': float(ft_urdu_metrics['accuracy']),
            'precision': float(ft_urdu_metrics['precision']),
            'recall': float(ft_urdu_metrics['recall']),
            'f1': float(ft_urdu_metrics['f1'])
        }
    }
    
    with open('results/week5_joint_training/comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Visualization
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Approach comparison
    ax1 = fig.add_subplot(gs[0, :])
    approaches = ['English Only\n(Zero-Shot)', 'Joint\nMultilingual', 'Fine-Tuned\non Urdu']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    x = np.arange(len(approaches))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [
            comparison['english_only'][metric.lower()],
            comparison['joint_multilingual'][metric.lower()],
            comparison['finetuned'][metric.lower()]
        ]
        ax1.bar(x + i*width - width*1.5, values, width, label=metric)
    
    ax1.set_ylabel('Score')
    ax1.set_title('Urdu Performance: Approach Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(approaches)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # 2-3. Confusion matrices
    cm_joint = confusion_matrix(joint_urdu_metrics['true_labels'], joint_urdu_metrics['predictions'])
    cm_ft = confusion_matrix(ft_urdu_metrics['true_labels'], ft_urdu_metrics['predictions'])
    
    cms = [cm_joint, cm_ft]
    titles = ['Joint Multilingual', 'Fine-Tuned']
    
    for idx, (cm, title) in enumerate(zip(cms, titles)):
        ax = fig.add_subplot(gs[1, idx])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
        ax.set_title(f'{title} - Urdu', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    # Summary table
    ax_summary = fig.add_subplot(gs[1, 2])
    ax_summary.axis('off')
    
    summary_text = "Performance Summary (Urdu)\n\n"
    summary_text += f"{'Approach':<20} {'Acc':<7} {'F1':<7}\n"
    summary_text += "-" * 35 + "\n"
    summary_text += f"{'Zero-Shot':<20} {comparison['english_only']['accuracy']:<7.4f} {comparison['english_only']['f1']:<7.4f}\n"
    summary_text += f"{'Joint Multilingual':<20} {comparison['joint_multilingual']['accuracy']:<7.4f} {comparison['joint_multilingual']['f1']:<7.4f}\n"
    summary_text += f"{'Fine-Tuned':<20} {comparison['finetuned']['accuracy']:<7.4f} {comparison['finetuned']['f1']:<7.4f}\n\n"
    
    improvements = {
        'joint_vs_zero': comparison['joint_multilingual']['accuracy'] - comparison['english_only']['accuracy'],
        'ft_vs_zero': comparison['finetuned']['accuracy'] - comparison['english_only']['accuracy']
    }
    
    summary_text += "Improvements over Zero-Shot:\n"
    summary_text += f"  Joint: +{improvements['joint_vs_zero']:.4f}\n"
    summary_text += f"  Fine-Tuned: +{improvements['ft_vs_zero']:.4f}\n"
    
    ax_summary.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                   verticalalignment='center', transform=ax_summary.transAxes)
    
    plt.savefig('results/week5_joint_training/comprehensive_comparison.png',
                dpi=300, bbox_inches='tight')
    print(f"  ✓ Visualizations saved")
    
    # Final summary
    print("\n" + "="*60)
    print("✅ Week 5 - Joint Training & Fine-Tuning Complete!")
    print("="*60)
    print("\nKey Results (Urdu Performance):")
    print(f"  ✔ Zero-Shot: {comparison['english_only']['accuracy']:.4f}")
    print(f"  ✔ Joint Multilingual: {comparison['joint_multilingual']['accuracy']:.4f} (+{improvements['joint_vs_zero']:.4f})")
    print(f"  ✔ Fine-Tuned: {comparison['finetuned']['accuracy']:.4f} (+{improvements['ft_vs_zero']:.4f})")
    print("\nNext: Week 6 - Final Evaluation & Documentation")

if __name__ == "__main__":
    main()
