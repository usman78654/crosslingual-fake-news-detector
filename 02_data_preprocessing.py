"""
Week 2: Data Preprocessing Pipeline
Cleans, tokenizes, and prepares data for model training
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import os
import pickle

def clean_text(text):
    """Basic text cleaning"""
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_and_preprocess(csv_path, language):
    """Load and preprocess dataset"""
    print(f"\nLoading {language} dataset from {csv_path}...")
    # Try different encodings
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except (UnicodeDecodeError, Exception):
            continue
    
    if df is None:
        raise ValueError(f"Could not load {csv_path} with any supported encoding")
    
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Identify text column
    text_col = None
    for col in df.columns:
        if col.lower() in ['text', 'title', 'news', 'news items', 'content', 'article']:
            text_col = col
            break
    
    if text_col is None:
        # Try to find column with string data
        for col in df.columns:
            if df[col].dtype == 'object' and col.lower() not in ['label', 'sr. no.', 'sr. no', 'index']:
                text_col = col
                break
    
    if text_col is None:
        raise ValueError(f"Could not identify text column in {csv_path}")
    
    print(f"  Using text column: '{text_col}'")
    
    # Extract text and labels
    df['text'] = df[text_col].apply(clean_text)
    
    # Remove empty texts
    df = df[df['text'].str.len() > 10]
    
    print(f"  After cleaning: {len(df)} samples")
    return df

def create_baseline_model(X_train, y_train, X_test, y_test, dataset_name):
    """Train baseline Logistic Regression model with TF-IDF"""
    print(f"\n{'='*60}")
    print(f"Training Baseline Model on {dataset_name}")
    print('='*60)
    
    # TF-IDF Vectorization
    print("\n[1/3] Applying TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"  Feature matrix shape: {X_train_tfidf.shape}")
    
    # Train Logistic Regression
    print("\n[2/3] Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    print("\n[3/3] Evaluating...")
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{dataset_name} Baseline Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'True']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return model, vectorizer, accuracy, cm

def main():
    print("\n" + "="*60)
    print("CROSS-LINGUAL FAKE NEWS DETECTION")
    print("Week 2: Data Preprocessing & Baseline")
    print("="*60)
    
    # Create directories
    os.makedirs('results/week2_preprocessing', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Load English Dataset
    print("\n[1/4] Processing English Dataset...")
    try:
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        eng_fake = None
        eng_true = None
        
        for encoding in encodings:
            try:
                eng_fake = pd.read_csv('EnglishDataset/Fake.csv', encoding=encoding)
                eng_true = pd.read_csv('EnglishDataset/True.csv', encoding=encoding)
                print(f"  Loaded with encoding: {encoding}")
                break
            except:
                continue
        
        eng_fake['label'] = 0
        eng_true['label'] = 1
        
        english_df = pd.concat([eng_fake, eng_true], ignore_index=True)
        
        # Find text column
        text_col = None
        for col in english_df.columns:
            if col.lower() in ['text', 'title', 'news', 'content']:
                text_col = col
                break
        
        if text_col is None:
            for col in english_df.columns:
                if english_df[col].dtype == 'object' and col.lower() != 'label':
                    text_col = col
                    break
        
        print(f"  Using column: {text_col}")
        english_df['text'] = english_df[text_col].apply(clean_text)
        english_df = english_df[english_df['text'].str.len() > 10]
        english_df = english_df[['text', 'label']]
        english_df['language'] = 'english'
        
        print(f"  Total samples: {len(english_df)}")
        print(f"  Fake: {(english_df['label']==0).sum()}")
        print(f"  True: {(english_df['label']==1).sum()}")
        
        # Save processed data
        english_df.to_csv('data/processed/english_processed.csv', index=False)
        print(f"  ✓ Saved to data/processed/english_processed.csv")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        english_df = None
    
    # Load Urdu Dataset
    print("\n[2/4] Processing Urdu Dataset...")
    try:
        # Try tab delimiter first for Urdu datasets
        delimiters = ['\t', ',', ';']
        urdu_fake = None
        urdu_true = None
        
        for delimiter in delimiters:
            try:
                urdu_fake = pd.read_csv('Urdu_Dataset/Fake News.csv', delimiter=delimiter)
                urdu_true = pd.read_csv('Urdu_Dataset/True News.csv', delimiter=delimiter)
                if len(urdu_fake.columns) > 1:  # Successfully parsed multiple columns
                    print(f"  Loaded with delimiter: {repr(delimiter)}")
                    break
            except:
                continue
        
        urdu_fake['label'] = 0
        urdu_true['label'] = 1
        
        urdu_df = pd.concat([urdu_fake, urdu_true], ignore_index=True)
        
        # Find text column
        text_col = None
        for col in urdu_df.columns:
            if 'news' in col.lower() or 'text' in col.lower():
                text_col = col
                break
        
        print(f"  Using column: {text_col}")
        urdu_df['text'] = urdu_df[text_col].apply(clean_text)
        urdu_df = urdu_df[urdu_df['text'].str.len() > 10]
        urdu_df = urdu_df[['text', 'label']]
        urdu_df['language'] = 'urdu'
        
        print(f"  Total samples: {len(urdu_df)}")
        print(f"  Fake: {(urdu_df['label']==0).sum()}")
        print(f"  True: {(urdu_df['label']==1).sum()}")
        
        # Save processed data
        urdu_df.to_csv('data/processed/urdu_processed.csv', index=False)
        print(f"  ✓ Saved to data/processed/urdu_processed.csv")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        urdu_df = None
    
    # Train-Test Split
    print("\n[3/4] Creating train-test splits...")
    results = {}
    
    if english_df is not None:
        X_eng_train, X_eng_test, y_eng_train, y_eng_test = train_test_split(
            english_df['text'], english_df['label'], 
            test_size=0.2, random_state=42, stratify=english_df['label']
        )
        print(f"  English - Train: {len(X_eng_train)}, Test: {len(X_eng_test)}")
        
        eng_train = pd.DataFrame({'text': X_eng_train, 'label': y_eng_train})
        eng_test = pd.DataFrame({'text': X_eng_test, 'label': y_eng_test})
        eng_train.to_csv('data/processed/english_train.csv', index=False)
        eng_test.to_csv('data/processed/english_test.csv', index=False)
        
        results['english'] = (X_eng_train, y_eng_train, X_eng_test, y_eng_test)
    
    if urdu_df is not None:
        X_urd_train, X_urd_test, y_urd_train, y_urd_test = train_test_split(
            urdu_df['text'], urdu_df['label'],
            test_size=0.2, random_state=42, stratify=urdu_df['label']
        )
        print(f"  Urdu - Train: {len(X_urd_train)}, Test: {len(X_urd_test)}")
        
        urd_train = pd.DataFrame({'text': X_urd_train, 'label': y_urd_train})
        urd_test = pd.DataFrame({'text': X_urd_test, 'label': y_urd_test})
        urd_train.to_csv('data/processed/urdu_train.csv', index=False)
        urd_test.to_csv('data/processed/urdu_test.csv', index=False)
        
        results['urdu'] = (X_urd_train, y_urd_train, X_urd_test, y_urd_test)
    
    # Baseline Models
    print("\n[4/4] Training baseline models...")
    baseline_results = {}
    confusion_matrices = {}
    
    if 'english' in results:
        model, vec, acc, cm = create_baseline_model(
            *results['english'], 'English'
        )
        baseline_results['english'] = acc
        confusion_matrices['english'] = cm
        
        # Save model
        with open('results/week2_preprocessing/baseline_english_model.pkl', 'wb') as f:
            pickle.dump((model, vec), f)
    
    if 'urdu' in results:
        model, vec, acc, cm = create_baseline_model(
            *results['urdu'], 'Urdu'
        )
        baseline_results['urdu'] = acc
        confusion_matrices['urdu'] = cm
        
        # Save model
        with open('results/week2_preprocessing/baseline_urdu_model.pkl', 'wb') as f:
            pickle.dump((model, vec), f)
    
    # Visualization
    if baseline_results:
        print("\n[5/5] Creating visualizations...")
        
        # Baseline comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        languages = list(baseline_results.keys())
        accuracies = list(baseline_results.values())
        
        axes[0].bar(languages, accuracies, color=['#3498db', '#e74c3c'])
        axes[0].set_title('Baseline Model Performance', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(accuracies):
            axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
        
        # Confusion matrices
        for idx, (lang, cm) in enumerate(confusion_matrices.items()):
            if len(confusion_matrices) == 1:
                ax = axes[1]
            else:
                if idx == 0:
                    continue
                ax = axes[1]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Fake', 'True'],
                       yticklabels=['Fake', 'True'])
            ax.set_title(f'{lang.title()} Confusion Matrix', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('results/week2_preprocessing/baseline_results.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved visualization")
    
    # Summary
    print("\n" + "="*60)
    print("✅ Week 2 - Data Preprocessing Complete!")
    print("="*60)
    print("\nDeliverables:")
    print("  ✔ Processed datasets created")
    print("  ✔ Train-test splits saved")
    print("  ✔ Baseline models trained")
    if baseline_results:
        for lang, acc in baseline_results.items():
            print(f"  ✔ {lang.title()} baseline accuracy: {acc:.4f}")
    print("\nNext: Week 3 - XLM-RoBERTa Implementation")

if __name__ == "__main__":
    main()
