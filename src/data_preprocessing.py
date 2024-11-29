import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

def clean_text(text):
    """
    Cleans the text by removing URLs, special characters, and stopwords (in English).

    Args:
        text (str): Original text.

    Returns:
        str: Cleaned text.
    """
    # Remove URLs and special characters/numbers, and convert to lowercase
    text = re.sub(r"http\S+|www.\S+|[^a-zA-Z\s]", "", text).lower()
    
    # Remove English stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in text.split() if word not in stop_words]
    
    return " ".join(filtered_tokens)

def extract_content(content):
    """
    Extracts main information from the content.

    Args:
        content (dict): Dictionary containing article information.

    Returns:
        tuple: (claim, clean_body, label)
    """
    claim = content.get("claim", "")
    body_text = " ".join(content.get("body-text", []))
    rating = content.get("rating", [[]])[0][2] if content.get("rating") else "Unrated"
    clean_body = clean_text(body_text)
    label = 1 if "true" in rating.lower() else 0
    return claim, clean_body, label

def preprocess_articles(input_file, output_file):
    """
    Preprocesses the data contained in the JSON news file and saves it as CSV.

    Args:
        input_file (str): Path to the raw JSON file.
        output_file (str): Path to save the processed CSV file.
    """
    # Load data from JSON
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    rows = []
    for url, content in data.items():
        try:
            claim, clean_body, label = extract_content(content)
            rows.append({"claim": claim, "body_text": clean_body, "label": label})
        except Exception as e:
            print(f"Error processing {url}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Split into train, validation, and test sets
    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])
    
    # Save the data
    train.to_csv(output_file.replace(".csv", "_train.csv"), index=False)
    val.to_csv(output_file.replace(".csv", "_val.csv"), index=False)
    test.to_csv(output_file.replace(".csv", "_test.csv"), index=False)
    print(f"Data saved to {output_file}")