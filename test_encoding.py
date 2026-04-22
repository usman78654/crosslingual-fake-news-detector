import pandas as pd

# Test different encodings
encodings_to_try = [
    'utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1', 
    'cp1256',  # Arabic/Urdu
    'utf-16',  # Another common encoding
    'windows-1256'  # Windows Arabic
]

file_path = 'Urdu_Dataset/Fake News.csv'

for encoding in encodings_to_try:
    try:
        df = pd.read_csv(file_path, encoding=encoding, nrows=5)
        print(f"✓ SUCCESS with encoding: {encoding}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  First row sample: {df.iloc[0, 0] if len(df) > 0 else 'N/A'}")
        print()
        break
    except Exception as e:
        print(f"✗ FAILED with {encoding}: {str(e)[:100]}")

print("\nTrying with error handling...")
try:
    df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
    print(f"✓ Loaded with utf-8 + errors='ignore'")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"✗ Failed: {e}")
