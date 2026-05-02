import os
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── File paths ─────────────────────────────────────────
# Defined once at the top so if paths ever change
# we only update them in one place not scattered
# throughout the code
RAW_DATA_PATH = 'data/raw/spam.csv'
PROCESSED_DATA_PATH = 'data/processed/spam_cleaned.csv'


# ── Text cleaning function ─────────────────────────────
def clean_text(text):
    """
    Cleans a single email message.

    From EDA we found:
    - Text has capitals → lowercase everything
    - Text has punctuation → remove it
    - Text has stopwords → remove them
    - Words have different forms → stem to root

    Input:  "CONGRATULATIONS!! You've WON a FREE prize!"
    Output: "congratul won free prize"
    """

    # Step 1 — lowercase
    # "FREE" and "free" are the same word
    # without this the model treats them as different
    text = text.lower()

    # Step 2 — remove punctuation
    # string.punctuation = !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    # We keep a character only if it is NOT in that string
    text = ''.join(
        char for char in text
        if char not in string.punctuation
    )

    # Step 3 — split into individual words
    # "free entry win" → ["free", "entry", "win"]
    words = text.split()

    # Step 4 — remove stopwords
    # From EDA we saw stopwords like "to", "a", "the"
    # appearing heavily in both spam and ham
    # They give the model no useful signal so we remove them
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Step 5 — stemming
    # From EDA we saw words like "winning", "winner", "wins"
    # all meaning the same thing
    # Stemming converts them all to the same root: "win"
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Step 6 — join back into one string
    # ["congratul", "won", "free", "prize"] → "congratul won free prize"
    return ' '.join(words)


# ── Main preprocessing function ────────────────────────
def preprocess():
    """
    Full preprocessing pipeline.

    Loads raw data → cleans it → saves processed version.
    Based on findings from notebooks/eda.ipynb
    """

    print("=" * 50)
    print("Starting preprocessing pipeline")
    print("=" * 50)

    # ── Step 1: Load raw data ───────────────────────────
    print("\n[1/7] Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH, encoding='latin-1')
    print(f"      Loaded {len(df)} rows, {len(df.columns)} columns")

    # ── Step 2: Drop useless columns ───────────────────
    # From EDA: columns 3, 4, 5 are completely empty
    # Only v1 (label) and v2 (text) have actual data
    print("\n[2/7] Dropping empty columns...")
    df = df[['v1', 'v2']]
    print(f"      Kept columns: {df.columns.tolist()}")

    # ── Step 3: Rename columns ──────────────────────────
    # v1 and v2 are meaningless names
    # label and text are immediately clear
    print("\n[3/7] Renaming columns...")
    df.columns = ['label', 'text']

    # ── Step 4: Convert labels to numbers ──────────────
    # Models need numbers not strings
    # spam = 1, ham = 0
    print("\n[4/7] Encoding labels...")
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    print(f"      Spam: {df['label'].sum()}")
    print(f"      Ham:  {(df['label'] == 0).sum()}")

    # ── Step 5: Drop duplicates ─────────────────────────
    # From EDA: we found duplicate messages
    # Duplicates in both train and test sets make the
    # model look better than it actually is
    print("\n[5/7] Removing duplicates...")
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"      Removed {before - after} duplicate rows")
    print(f"      Remaining: {after} rows")

    # ── Step 6: Add length feature ──────────────────────
    # From EDA: spam messages are consistently longer
    # than ham messages — this is useful signal for the model
    print("\n[6/7] Adding length feature...")
    df['length'] = df['text'].apply(len)
    print(f"      Average spam length:  {df[df['label']==1]['length'].mean():.0f} chars")
    print(f"      Average ham length:   {df[df['label']==0]['length'].mean():.0f} chars")

    # ── Step 7: Clean text ──────────────────────────────
    # Lowercase, remove punctuation, remove stopwords, stem
    print("\n[7/7] Cleaning text...")
    df['text'] = df['text'].apply(clean_text)
    print("      Text cleaning complete")

    # ── Save processed data ─────────────────────────────
    # Create the folder if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("\n" + "=" * 50)
    print("Preprocessing complete")
    print(f"Saved to: {PROCESSED_DATA_PATH}")
    print(f"Final shape: {df.shape}")
    print("=" * 50)

    return df


# ── Entry point ─────────────────────────────────────────
# Runs only when you execute: python src/preprocess.py
# Does NOT run when train.py imports this function
if __name__ == '__main__':
    preprocess()