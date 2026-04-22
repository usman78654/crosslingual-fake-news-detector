import pandas as pd

# Load and clean Urdu datasets
print("Loading Urdu datasets...")

# Try different encoding/delimiter combinations
try:
    fake_df = pd.read_csv('Urdu_Dataset/Fake News.csv', delimiter='\t', encoding='latin-1')
    print("Fake loaded with latin-1, tab delimiter")
except:
    try:
        fake_df = pd.read_csv('Urdu_Dataset/Fake News.csv', delimiter='\t', encoding='cp1252')
        print("Fake loaded with cp1252, tab delimiter")
    except:
        fake_df = pd.read_csv('Urdu_Dataset/Fake News.csv', encoding='latin-1')
        print("Fake loaded with latin-1, comma delimiter")

try:
    true_df = pd.read_csv('Urdu_Dataset/True News.csv', delimiter=',', encoding='utf-8')
    print("True loaded with utf-8, comma delimiter")  
except:
    try:
        true_df = pd.read_csv('Urdu_Dataset/True News.csv', delimiter=',', encoding='latin-1')
        print("True loaded with latin-1, comma delimiter")
    except:
        true_df = pd.read_csv('Urdu_Dataset/True News.csv', delimiter='\t', encoding='latin-1')
        print("True loaded with latin-1, tab delimiter")

print(f"\nFake columns: {fake_df.columns.tolist()}")
print(f"True columns: {true_df.columns.tolist()}")

# Clean fake dataset
if 'News Items' in fake_df.columns:
    fake_text = fake_df['News Items']
elif len(fake_df.columns) >= 2:
    fake_text = fake_df.iloc[:, 1]  # Second column
else:
    fake_text = fake_df.iloc[:, 0]

fake_clean = pd.DataFrame({
    'text': fake_text,
    'label': 0
})

# Clean true dataset  
if 'News Items' in true_df.columns:
    true_text = true_df['News Items']
elif len(true_df.columns) >= 2:
    true_text = true_df.iloc[:, 1]  # Second column
else:
    true_text = true_df.iloc[:, 0]

true_clean = pd.DataFrame({
    'text': true_text,
    'label': 1
})

# Combine
urdu_combined = pd.concat([fake_clean, true_clean], ignore_index=True)
urdu_combined['text'] = urdu_combined['text'].astype(str).str.strip()
urdu_combined = urdu_combined[urdu_combined['text'].str.len() > 10]

print(f"\nUrdu combined: {len(urdu_combined)} samples")
print(f"Fake: {(urdu_combined['label']==0).sum()}")
print(f"True: {(urdu_combined['label']==1).sum()}")

# Save
urdu_combined.to_csv('data/processed/urdu_processed.csv', index=False)
print("\n✓ Saved cleaned Urdu dataset")

# Create train-test split
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    urdu_combined, test_size=0.2, random_state=42, stratify=urdu_combined['label']
)

train_df.to_csv('data/processed/urdu_train.csv', index=False)
test_df.to_csv('data/processed/urdu_test.csv', index=False)

print(f"Train: {len(train_df)}, Test: {len(test_df)}")
print("✓ Saved Urdu train/test splits")
