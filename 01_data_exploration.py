"""
Week 1: Data Exploration and Analysis
This script explores the English and Urdu datasets
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import re

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def normalize_column_name(col_name):
    """Normalize column names for robust matching."""
    clean = str(col_name).replace('\ufeff', '').strip().lower()
    clean = re.sub(r'\s+', ' ', clean)
    return clean


def read_csv_with_fallback(file_path, read_options, encodings):
    """Try multiple encoding + read options and return first valid parse."""
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


def print_text_quality(df, text_column, dataset_name):
    """Print lightweight text quality checks to surface corrupted inputs."""
    text_values = df[text_column].astype(str)
    lengths = text_values.str.len().clip(lower=1)
    question_density = text_values.str.count(r'\?') / lengths
    control_chars = text_values.str.contains(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', regex=True)

    heavy_question_rows = (question_density > 0.5).mean() * 100
    control_char_rows = control_chars.mean() * 100

    print(f"\nText quality checks ({dataset_name}):")
    print(f"  Rows with >50% '?' characters: {heavy_question_rows:.2f}%")
    print(f"  Rows with control characters: {control_char_rows:.2f}%")
    if heavy_question_rows > 20:
        print("  WARNING: High '?' ratio detected. Source text may be partially corrupted.")


def explore_dataset(
    fake_path,
    true_path,
    dataset_name,
    fake_read_options,
    true_read_options,
    fake_encodings,
    true_encodings,
):
    """Explore and visualize dataset statistics"""
    print(f"\n{'='*60}")
    print(f"{dataset_name} Dataset Exploration")
    print('='*60)
    
    # Load datasets
    print(f"\nLoading {dataset_name} datasets...")
    fake_df, fake_encoding, fake_options = read_csv_with_fallback(
        fake_path,
        read_options=fake_read_options,
        encodings=fake_encodings,
    )
    true_df, true_encoding, true_options = read_csv_with_fallback(
        true_path,
        read_options=true_read_options,
        encodings=true_encodings,
    )

    print(
        f"  Fake file loaded with encoding={fake_encoding}, "
        f"options={fake_options}"
    )
    print(
        f"  True file loaded with encoding={true_encoding}, "
        f"options={true_options}"
    )
    
    # Add labels
    fake_df['label'] = 0  # Fake
    true_df['label'] = 1  # True
    
    # Combine
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    print(f"\n{dataset_name} Dataset Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Fake news: {len(fake_df)} ({len(fake_df)/len(df)*100:.2f}%)")
    print(f"  True news: {len(true_df)} ({len(true_df)/len(df)*100:.2f}%)")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head(3))
    
    # Missing values
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Text length analysis
    text_column = select_text_column(df, ['text', 'news items', 'news', 'title', 'content'])
    
    if text_column:
        df['text_length'] = df[text_column].astype(str).str.len()
        print(f"\nText length statistics ({text_column}):")
        print(df.groupby('label')['text_length'].describe())
        print_text_quality(df, text_column, dataset_name)
    
    return df

def main():
    print("\n" + "="*60)
    print("CROSS-LINGUAL FAKE NEWS DETECTION")
    print("Data Exploration Phase")
    print("="*60)
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/week1_exploration', exist_ok=True)
    
    # English Dataset
    print("\n[1/2] Exploring English Dataset...")
    try:
        english_df = explore_dataset(
            'EnglishDataset/Fake.csv',
            'EnglishDataset/True.csv',
            'English',
            fake_read_options=[{'delimiter': ','}],
            true_read_options=[{'delimiter': ','}],
            fake_encodings=['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1'],
            true_encodings=['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1'],
        )
        
        # Save combined English dataset
        english_df['language'] = 'english'
        english_df.to_csv('results/week1_exploration/english_combined.csv', index=False)
        print(f"\n✓ Saved combined English dataset to results/week1_exploration/english_combined.csv")
        
    except Exception as e:
        print(f"✗ Error loading English dataset: {e}")
        english_df = None
    
    # Urdu Dataset
    print("\n[2/2] Exploring Urdu Dataset...")
    try:
        urdu_df = explore_dataset(
            'Urdu_Dataset/Fake News.csv',
            'Urdu_Dataset/True News.csv',
            'Urdu',
            fake_read_options=[{'delimiter': '\t'}],
            true_read_options=[{'delimiter': ','}],
            fake_encodings=['cp1256', 'latin-1', 'cp1252', 'iso-8859-1'],
            true_encodings=['utf-8-sig', 'utf-8', 'cp1256', 'latin-1'],
        )
        
        # Save combined Urdu dataset
        urdu_df['language'] = 'urdu'
        urdu_df.to_csv('results/week1_exploration/urdu_combined.csv', index=False)
        print(f"\n✓ Saved combined Urdu dataset to results/week1_exploration/urdu_combined.csv")
        
    except Exception as e:
        print(f"✗ Error loading Urdu dataset: {e}")
        urdu_df = None
    
    # Visualization
    if english_df is not None or urdu_df is not None:
        print("\n[3/3] Creating visualizations...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        if english_df is not None:
            english_counts = english_df['label'].value_counts()
            axes[0].bar(['Fake', 'True'], [english_counts[0], english_counts[1]], 
                       color=['#e74c3c', '#2ecc71'])
            axes[0].set_title('English Dataset Distribution', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Count')
            axes[0].grid(axis='y', alpha=0.3)
        
        if urdu_df is not None:
            urdu_counts = urdu_df['label'].value_counts()
            axes[1].bar(['Fake', 'True'], [urdu_counts[0], urdu_counts[1]], 
                       color=['#e74c3c', '#2ecc71'])
            axes[1].set_title('Urdu Dataset Distribution', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Count')
            axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/week1_exploration/dataset_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to results/week1_exploration/dataset_distribution.png")
    
    print("\n" + "="*60)
    print("✅ Week 1 - Data Exploration Complete!")
    print("="*60)
    print("\nDeliverables:")
    print("  ✔ Dataset statistics analyzed")
    print("  ✔ Combined datasets saved")
    print("  ✔ Distribution visualizations created")
    print("\nNext: Week 2 - Data Preprocessing")

if __name__ == "__main__":
    main()
