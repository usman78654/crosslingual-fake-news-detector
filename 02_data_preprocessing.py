"""
Week 2: Data Preprocessing Pipeline
Cleans, tokenizes, and prepares data for model training
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import os
import pickle


def normalize_column_name(col_name):
    """Normalize column names for robust matching."""
    clean = str(col_name).replace('\ufeff', '').strip().lower()
    clean = re.sub(r'\s+', ' ', clean)
    return clean


def read_csv_with_fallback(file_path, read_options, encodings):
    """Try multiple encoding + read-option combinations."""
    last_error = None

    for options in read_options:
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, **options)
                if len(df.columns) <= 1:
                    continue

                df.columns = [str(col).replace('\ufeff', '').strip() for col in df.columns]
                return df, encoding, options
            except Exception as exc:
                last_error = exc

    raise ValueError(
        f"Could not parse {file_path} with provided encodings/options. "
        f"Last error: {last_error}"
    )


def select_text_column(df, preferred_columns):
    """Pick text column by preference, then fallback to object columns."""
    normalized = {normalize_column_name(col): col for col in df.columns}

    for preferred in preferred_columns:
        candidate = normalized.get(normalize_column_name(preferred))
        if candidate is not None:
            return candidate

    excluded = {
        'label', 'sr. no.', 'sr. no', 'sr no', 'sr_no', 'index', 'date', 'subject'
    }
    for col in df.columns:
        if df[col].dtype == 'object' and normalize_column_name(col) not in excluded:
            return col

    return None

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


def print_text_quality(df, text_column, dataset_name):
    """Print data-quality diagnostics for cleaned text."""
    text_values = df[text_column].astype(str)
    lengths = text_values.str.len().clip(lower=1)
    question_density = text_values.str.count(r'\?') / lengths
    control_chars = text_values.str.contains(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', regex=True)

    heavy_question_rows = (question_density > 0.5).mean() * 100
    control_char_rows = control_chars.mean() * 100

    print(f"  Text quality ({dataset_name}):")
    print(f"    Rows with >50% '?' characters: {heavy_question_rows:.2f}%")
    print(f"    Rows with control characters: {control_char_rows:.2f}%")
    if heavy_question_rows > 20:
        print("    WARNING: High '?' ratio detected. Source text may be partially corrupted.")


def validate_label_distribution(df, dataset_name):
    """Ensure both classes are present before split/training."""
    class_count = df['label'].nunique()
    if class_count < 2:
        raise ValueError(
            f"{dataset_name} has {class_count} class after preprocessing. "
            "Need at least 2 classes for stratified split and classification."
        )

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
    print(classification_report(y_test, y_pred, target_names=['Fake', 'True'], zero_division=0))
    
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
        eng_fake, fake_encoding, _ = read_csv_with_fallback(
            'EnglishDataset/Fake.csv',
            read_options=[{'delimiter': ','}],
            encodings=['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1'],
        )
        eng_true, true_encoding, _ = read_csv_with_fallback(
            'EnglishDataset/True.csv',
            read_options=[{'delimiter': ','}],
            encodings=['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1'],
        )
        print(f"  Fake loaded with encoding: {fake_encoding}")
        print(f"  True loaded with encoding: {true_encoding}")
        
        eng_fake['label'] = 0
        eng_true['label'] = 1
        
        english_df = pd.concat([eng_fake, eng_true], ignore_index=True)
        
        # Prefer article body over title for more realistic fake-news detection.
        text_col = select_text_column(
            english_df,
            preferred_columns=['text', 'content', 'article', 'title', 'news'],
        )
        if text_col is None:
            raise ValueError("Could not identify text column in English dataset")
        
        print(f"  Using column: {text_col}")
        english_df['text'] = english_df[text_col].apply(clean_text)
        english_df = english_df[english_df['text'].str.len() > 10]
        english_df = english_df[['text', 'label']]
        english_df['language'] = 'english'

        print_text_quality(english_df, 'text', 'English')
        validate_label_distribution(english_df, 'English dataset')
        
        print(f"  Total samples: {len(english_df)}")
        print(f"  Fake: {(english_df['label']==0).sum()}")
        print(f"  True: {(english_df['label']==1).sum()}")
        
        # IMPORTANT: Remove duplicates BEFORE train/test split to prevent data leakage
        print(f"  Removing duplicate texts...")
        before_dedup = len(english_df)
        english_df = english_df.drop_duplicates(subset=['text'], keep='first')
        after_dedup = len(english_df)
        print(f"  Removed {before_dedup - after_dedup} duplicate texts")
        print(f"  Samples after deduplication: {len(english_df)}")
        
        # Save processed data
        english_df.to_csv('data/processed/english_processed.csv', index=False)
        print(f"  ✓ Saved to data/processed/english_processed.csv")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        english_df = None
    
    # Load Urdu Dataset
    print("\n[2/4] Processing Urdu Dataset...")
    try:
        # Urdu files use different formats by source.
        urdu_fake, fake_encoding, _ = read_csv_with_fallback(
            'Urdu_Dataset/Fake News.csv',
            read_options=[{'delimiter': '\t'}],
            encodings=['cp1256', 'latin-1', 'cp1252', 'iso-8859-1'],
        )
        urdu_true, true_encoding, _ = read_csv_with_fallback(
            'Urdu_Dataset/True News.csv',
            read_options=[{'delimiter': ','}],
            encodings=['utf-8-sig', 'utf-8', 'cp1256', 'latin-1'],
        )
        print(f"  Fake News encoding: {fake_encoding}")
        print(f"  True News encoding: {true_encoding}")
        
        print(f"  Fake News loaded: {urdu_fake.shape}, cols: {urdu_fake.columns.tolist()}")
        print(f"  True News loaded: {urdu_true.shape}, cols: {urdu_true.columns.tolist()}")
        
        urdu_fake['label'] = 0
        urdu_true['label'] = 1
        
        urdu_df = pd.concat([urdu_fake, urdu_true], ignore_index=True)
        
        text_col = select_text_column(
            urdu_df,
            preferred_columns=['news items', 'text', 'news', 'content', 'title'],
        )
        if text_col is None:
            raise ValueError("Could not identify text column in Urdu dataset")
        
        print(f"  Using column: {text_col}")
        urdu_df['text'] = urdu_df[text_col].apply(clean_text)
        urdu_df = urdu_df[urdu_df['text'].str.len() > 10]
        urdu_df = urdu_df[['text', 'label']]
        urdu_df['language'] = 'urdu'

        print_text_quality(urdu_df, 'text', 'Urdu')
        validate_label_distribution(urdu_df, 'Urdu dataset')
        
        print(f"  Total samples: {len(urdu_df)}")
        print(f"  Fake: {(urdu_df['label']==0).sum()}")
        print(f"  True: {(urdu_df['label']==1).sum()}")
        
        # IMPORTANT: Remove duplicates BEFORE train/test split to prevent data leakage
        print(f"  Removing duplicate texts...")
        before_dedup = len(urdu_df)
        urdu_df = urdu_df.drop_duplicates(subset=['text'], keep='first')
        after_dedup = len(urdu_df)
        print(f"  Removed {before_dedup - after_dedup} duplicate texts")
        print(f"  Samples after deduplication: {len(urdu_df)}")
        
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
