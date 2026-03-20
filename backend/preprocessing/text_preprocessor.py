"""
Text Preprocessing Module
=========================
Handles cleaning and preprocessing of raw customer support ticket text.
- Lowercasing
- HTML/special character removal
- Stopword removal
- Lemmatization
- Combining subject + description into a single feature
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK data is downloaded
def download_nltk_data():
    """Download required NLTK data packages."""
    packages = ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']
    for pkg in packages:
        try:
            nltk.data.find(f'corpora/{pkg}' if pkg != 'punkt' and pkg != 'punkt_tab' else f'tokenizers/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)

download_nltk_data()

# Initialize once
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Clean a single text string:
    1. Lowercase
    2. Remove HTML tags
    3. Remove URLs
    4. Remove special characters and digits
    5. Remove extra whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove special characters, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_stopwords(text: str) -> str:
    """Remove English stopwords from text."""
    words = text.split()
    filtered = [w for w in words if w not in _stop_words]
    return ' '.join(filtered)


def lemmatize_text(text: str) -> str:
    """Lemmatize each word in the text."""
    words = text.split()
    lemmatized = [_lemmatizer.lemmatize(w) for w in words]
    return ' '.join(lemmatized)


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline for a single text string.
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text


def combine_text_fields(subject: str, description: str) -> str:
    """
    Combine ticket subject and description into a single text feature.
    """
    subject = str(subject) if subject else ""
    description = str(description) if description else ""
    combined = f"{subject} {description}"
    return combined.strip()


def preprocess_dataframe(df):
    """
    Preprocess an entire DataFrame of tickets.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'Ticket Subject' and 'Ticket Description' columns.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with added 'combined_text' and 'processed_text' columns.
    """
    import pandas as pd
    
    df = df.copy()
    
    # Fill NaN values
    df['Ticket Subject'] = df['Ticket Subject'].fillna('')
    df['Ticket Description'] = df['Ticket Description'].fillna('')
    
    # Combine subject and description
    df['combined_text'] = df.apply(
        lambda row: combine_text_fields(row['Ticket Subject'], row['Ticket Description']),
        axis=1
    )
    
    # Apply full preprocessing pipeline
    print("  → Cleaning and preprocessing text...")
    df['processed_text'] = df['combined_text'].apply(preprocess_text)
    
    # Remove rows with empty processed text
    empty_mask = df['processed_text'].str.strip() == ''
    if empty_mask.any():
        print(f"  → Removed {empty_mask.sum()} rows with empty text after preprocessing")
        df = df[~empty_mask].reset_index(drop=True)
    
    print(f"  → Preprocessed {len(df)} tickets successfully")
    return df


if __name__ == "__main__":
    # Quick test
    sample = "I'm having an issue with the {product_purchased}. Please assist. <br> Visit http://example.com"
    print(f"Original: {sample}")
    print(f"Cleaned:  {preprocess_text(sample)}")
