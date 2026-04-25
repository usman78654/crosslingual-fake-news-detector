"""Prepare clean Urdu train/test splits from the combined source file.

This utility avoids the corrupted standalone fake-news CSV and enforces
basic anti-skew checks before writing processed files.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

COMBINED_PATH = 'Urdu_Dataset/Combined .csv'
PROCESSED_PATH = 'data/processed/urdu_processed.csv'
TRAIN_PATH = 'data/processed/urdu_train.csv'
TEST_PATH = 'data/processed/urdu_test.csv'


def clean_text(text):
    if pd.isna(text):
        return ''
    return str(text).strip()


def map_binary_label(value):
    if pd.isna(value):
        return None

    raw = str(value).strip()
    if not raw:
        return None

    normalized = raw.upper()
    mapping = {
        'FAKE': 0,
        'FAKE NEWS': 0,
        'FALSE': 0,
        'TRUE': 1,
        'TRUE NEWS': 1,
        'REAL': 1,
    }

    if normalized in mapping:
        return mapping[normalized]

    try:
        numeric = float(raw)
        if numeric in (0.0, 1.0):
            return int(numeric)
    except ValueError:
        pass

    return None


def validate_artifact_skew(df, max_gap=0.2):
    stats = {}
    for label in (0, 1):
        subset = df[df['label'] == label]
        if len(subset) == 0:
            continue

        text_values = subset['text'].astype(str)
        lengths = text_values.str.len().clip(lower=1)
        question_density = text_values.str.count(r'\?') / lengths
        contains_question = text_values.str.contains(r'\?')

        stats[label] = {
            'contains_q': float(contains_question.mean()),
            'heavy_q': float((question_density > 0.5).mean()),
            'count': int(len(subset)),
        }

    if 0 in stats and 1 in stats:
        contains_gap = abs(stats[0]['contains_q'] - stats[1]['contains_q'])
        heavy_gap = abs(stats[0]['heavy_q'] - stats[1]['heavy_q'])

        print('Artifact check:')
        print(
            f"  Label 0 contains '?': {stats[0]['contains_q']*100:.2f}% | "
            f"Label 1 contains '?': {stats[1]['contains_q']*100:.2f}%"
        )
        print(
            f"  Label 0 heavy '?': {stats[0]['heavy_q']*100:.2f}% | "
            f"Label 1 heavy '?': {stats[1]['heavy_q']*100:.2f}%"
        )

        if contains_gap > max_gap or heavy_gap > max_gap:
            raise ValueError(
                'Detected strong label-correlated text artifacts. '
                'Dataset is not safe for model training.'
            )


def main():
    print('Loading clean Urdu combined dataset...')
    df = pd.read_csv(COMBINED_PATH, encoding='utf-8-sig')

    normalized_cols = {str(col).strip().lower(): col for col in df.columns}
    text_col = normalized_cols.get('news items')
    label_col = normalized_cols.get('label')

    if text_col is None or label_col is None:
        raise ValueError('Expected columns News Items and Label in combined Urdu file.')

    working = df.copy()
    working['text'] = working[text_col].apply(clean_text)
    working['label'] = working[label_col].apply(map_binary_label)

    valid_rows = working['label'].isin([0, 1]) & (working['text'].str.len() > 10)
    dropped_rows = int((~valid_rows).sum())
    if dropped_rows > 0:
        print(f'Dropped invalid/short rows: {dropped_rows}')

    working = working.loc[valid_rows, ['text', 'label']].copy()
    working['label'] = working['label'].astype(int)

    before_dedup = len(working)
    working = working.drop_duplicates(subset=['text'], keep='first')
    print(f'Removed duplicates: {before_dedup - len(working)}')

    validate_artifact_skew(working)

    print(f'Final Urdu samples: {len(working)}')
    print(f"Fake: {(working['label'] == 0).sum()}")
    print(f"True: {(working['label'] == 1).sum()}")

    working.to_csv(PROCESSED_PATH, index=False)

    train_df, test_df = train_test_split(
        working,
        test_size=0.2,
        random_state=42,
        stratify=working['label'],
    )

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f'Train: {len(train_df)} | Test: {len(test_df)}')
    print('Saved cleaned Urdu processed/train/test files.')


if __name__ == '__main__':
    main()
