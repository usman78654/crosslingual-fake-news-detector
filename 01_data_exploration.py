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

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def explore_dataset(fake_path, true_path, dataset_name):
    """Explore and visualize dataset statistics"""
    print(f"\n{'='*60}")
    print(f"{dataset_name} Dataset Exploration")
    print('='*60)
    
    # Load datasets
    print(f"\nLoading {dataset_name} datasets...")
    # Try different encodings and delimiters
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    delimiters = [',', '\t', ';']
    fake_df = None
    true_df = None
    
    for encoding in encodings:
        for delimiter in delimiters:
            try:
                fake_df = pd.read_csv(fake_path, encoding=encoding, delimiter=delimiter)
                true_df = pd.read_csv(true_path, encoding=encoding, delimiter=delimiter)
                # Check if we have multiple columns (successful parse)
                if len(fake_df.columns) > 1:
                    print(f"  Successfully loaded with encoding: {encoding}, delimiter: {repr(delimiter)}")
                    break
            except (UnicodeDecodeError, Exception):
                continue
        if fake_df is not None and len(fake_df.columns) > 1:
            break
    
    if fake_df is None or true_df is None:
        raise ValueError(f"Could not load datasets with any supported encoding")
    
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
    text_column = None
    for col in df.columns:
        if 'text' in col.lower() or 'news' in col.lower() or 'title' in col.lower():
            text_column = col
            break
    
    if text_column:
        df['text_length'] = df[text_column].astype(str).str.len()
        print(f"\nText length statistics ({text_column}):")
        print(df.groupby('label')['text_length'].describe())
    
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
            'English'
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
            'Urdu'
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
