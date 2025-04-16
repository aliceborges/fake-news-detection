import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
from enum import Enum, IntEnum


nltk.download('stopwords')


class BinaryLabel(IntEnum):
    """Enumeration for binary labels (truthfulness levels)"""
    FALSE = 0
    TRUE = 1


class MulticlassLabel(IntEnum):
    """Enumeration for multiclass labels (truthfulness levels)"""
    PANTS_ON_FIRE = 0
    FALSE = 1
    MOSTLY_FALSE = 2
    MIXED = 3
    MOSTLY_TRUE = 4
    TRUE = 5


class Rating(Enum):
    """Enumeration for different rating systems"""

    # Politifact
    TRUE = "True"
    MOSTLY_TRUE = "Mostly True"
    HALF_TRUE = "Half True"
    MOSTLY_FALSE = "Mostly False"
    FALSE = "False"
    PANTS_ON_FIRE = "Pants on Fire"
    
    # Snopes
    TRUE_SNOPES = "rating-true"
    MOSTLY_TRUE_SNOPES = "rating-mostly-true-new"
    MIXTURE = "rating-mixture"
    MOSTLY_FALSE_SNOPES = "rating-mostly-false"
    FALSE_SNOPES = "rating-false"
    UNPROVEN = "rating-unproven"
    UNFOUNDED = "rating-unfounded"
    OUTDATED = "rating-outdated"
    MISATTRIBUTED = "rating-misattributed"
    CORRECTLY_ATTRIBUTED = "rating-correctly-attributed"
    MISCAPTIONED = "rating-miscaptioned"
    LEGEND = "rating-legend"
    SCAM = "rating-scam"
    LEGIT = "rating-legit"
    LABELED_SATIRE = "rating-labeled-satire"
    ORIGINATED_AS_SATIRE = "rating-originated-as-satire"
    RECALL = "rating-recall"
    FAKE = "rating-fake"
    RESEARCH_IN_PROGRESS = "rating-research-in-progress"
    
    TRUE_POLI_ACT_TRUTH = "TRUE POLI ACT TRUTH - - METERM"
    POLITIF_TRUTH_O_TERM = "POLITIF TRUTH-O - TER'M"
    LITIFACT_TRUT_O_METERM = "LITIFACT TRUT - O-METER'M"
    FALSE_POLITIFACT_TRUTH_O_METER = "FALSE POLITIFACT TRUTH-O-METER ' \""
    PANTS_ON_FIRE_SLASH = "PANTS ON FIRE /"
    TRUE_POLITIFACT_TRUTH_O_METERM = "TRUE POLITIFACT TRUTH-O-METER'M"


def normalize_rating(original_rating: str) -> Rating:
    """Normalizes the rating to a standard format"""
    
    rating_map = {
        # PolitiFact original formats
        "TRUE POLI ACT TRUTH - - METERM": Rating.TRUE,
        "POLITIF TRUTH-O - TER'M": Rating.MOSTLY_TRUE,
        "LITIFACT TRUT - O-METER'M": Rating.MOSTLY_FALSE,
        "FALSE POLITIFACT TRUTH-O-METER ' \"": Rating.FALSE,
        "PANTS ON FIRE /": Rating.PANTS_ON_FIRE,
        "TRUE POLITIFACT TRUTH-O-METER'M": Rating.TRUE,
        
        # PolitiFact standard ratings
        "True": Rating.TRUE,
        "Mostly True": Rating.MOSTLY_TRUE,
        "Half True": Rating.HALF_TRUE,
        "Mostly False": Rating.MOSTLY_FALSE,
        "False": Rating.FALSE,
        "Pants on Fire": Rating.PANTS_ON_FIRE,
        
        # Snopes ratings
        "rating-true": Rating.TRUE_SNOPES,
        "rating-mostly-true-new": Rating.MOSTLY_TRUE_SNOPES,
        "rating-mixture": Rating.MIXTURE,
        "rating-mostly-false": Rating.MOSTLY_FALSE_SNOPES,
        "rating-false": Rating.FALSE_SNOPES,
        "rating-unproven": Rating.UNPROVEN,
        "rating-unfounded": Rating.UNFOUNDED,
        "rating-outdated": Rating.OUTDATED,
        "rating-misattributed": Rating.MISATTRIBUTED,
        "rating-correctly-attributed": Rating.CORRECTLY_ATTRIBUTED,
        "rating-miscaptioned": Rating.MISCAPTIONED,
        "rating-legend": Rating.LEGEND,
        "rating-scam": Rating.SCAM,
        "rating-legit": Rating.LEGIT,
        "rating-labeled-satire": Rating.LABELED_SATIRE,
        "rating-originated-as-satire": Rating.ORIGINATED_AS_SATIRE,
        "rating-recall": Rating.RECALL,
        "rating-fake": Rating.FAKE,
        "rating-research-in-progress": Rating.RESEARCH_IN_PROGRESS
    }
    
    return rating_map.get(original_rating, Rating.UNPROVEN)


def clean_text(text):
    """Clean the text by removing URLs, special characters, and stop words"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www.\S+|[^a-zA-Z\s]", "", text).lower()
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(filtered_tokens)


def rating_to_numeric(rating: Rating) -> tuple:
    """Convert rating to binary and multiclass labels"""
    
    binary_map = {
        # PolitiFact
        Rating.TRUE: BinaryLabel.TRUE,
        Rating.MOSTLY_TRUE: BinaryLabel.TRUE,
        Rating.HALF_TRUE: BinaryLabel.FALSE,
        Rating.MOSTLY_FALSE: BinaryLabel.FALSE,
        Rating.FALSE: BinaryLabel.FALSE,
        Rating.PANTS_ON_FIRE: BinaryLabel.FALSE,
        
        # Snopes
        Rating.TRUE_SNOPES: BinaryLabel.TRUE,
        Rating.MOSTLY_TRUE_SNOPES: BinaryLabel.TRUE,
        Rating.CORRECTLY_ATTRIBUTED: BinaryLabel.TRUE,
        Rating.LEGIT: BinaryLabel.TRUE,
        Rating.RECALL: BinaryLabel.TRUE,
        
        # Snopes
        Rating.MIXTURE: BinaryLabel.FALSE,
        Rating.MOSTLY_FALSE_SNOPES: BinaryLabel.FALSE,
        Rating.FALSE_SNOPES: BinaryLabel.FALSE,
        Rating.UNPROVEN: BinaryLabel.FALSE,
        Rating.UNFOUNDED: BinaryLabel.FALSE,
        Rating.OUTDATED: BinaryLabel.FALSE,
        Rating.MISATTRIBUTED: BinaryLabel.FALSE,
        Rating.MISCAPTIONED: BinaryLabel.FALSE,
        Rating.LEGEND: BinaryLabel.FALSE,
        Rating.SCAM: BinaryLabel.FALSE,
        Rating.LABELED_SATIRE: BinaryLabel.FALSE,
        Rating.ORIGINATED_AS_SATIRE: BinaryLabel.FALSE,
        Rating.FAKE: BinaryLabel.FALSE,
        Rating.RESEARCH_IN_PROGRESS: BinaryLabel.FALSE
    }
    
    multiclass_map = {
        # PolitiFact
        Rating.TRUE: MulticlassLabel.TRUE,
        Rating.MOSTLY_TRUE: MulticlassLabel.MOSTLY_TRUE,
        Rating.HALF_TRUE: MulticlassLabel.MIXED,
        Rating.MOSTLY_FALSE: MulticlassLabel.MOSTLY_FALSE,
        Rating.FALSE: MulticlassLabel.FALSE,
        Rating.PANTS_ON_FIRE: MulticlassLabel.PANTS_ON_FIRE,
        
        # Snopes
        Rating.TRUE_SNOPES: MulticlassLabel.TRUE,
        Rating.MOSTLY_TRUE_SNOPES: MulticlassLabel.MOSTLY_TRUE,
        Rating.CORRECTLY_ATTRIBUTED: MulticlassLabel.MOSTLY_TRUE,
        Rating.LEGIT: MulticlassLabel.MOSTLY_TRUE,
        Rating.RECALL: MulticlassLabel.MOSTLY_TRUE,
        Rating.MIXTURE: MulticlassLabel.MIXED,
        Rating.MOSTLY_FALSE_SNOPES: MulticlassLabel.MOSTLY_FALSE,
        Rating.FALSE_SNOPES: MulticlassLabel.FALSE,
        Rating.UNPROVEN: MulticlassLabel.FALSE,
        Rating.UNFOUNDED: MulticlassLabel.FALSE,
        Rating.OUTDATED: MulticlassLabel.FALSE,
        Rating.MISATTRIBUTED: MulticlassLabel.FALSE,
        Rating.MISCAPTIONED: MulticlassLabel.FALSE,
        Rating.LEGEND: MulticlassLabel.FALSE,
        Rating.SCAM: MulticlassLabel.PANTS_ON_FIRE,
        Rating.LABELED_SATIRE: MulticlassLabel.PANTS_ON_FIRE,
        Rating.ORIGINATED_AS_SATIRE: MulticlassLabel.PANTS_ON_FIRE,
        Rating.FAKE: MulticlassLabel.PANTS_ON_FIRE,
        Rating.RESEARCH_IN_PROGRESS: MulticlassLabel.FALSE
    }
    
    return binary_map.get(rating, BinaryLabel.FALSE), multiclass_map.get(rating, MulticlassLabel.FALSE)


def extract_content(content, origin):
    """Extract relevant content from the JSON data"""

    claim = content.get("claim", "")
    body_text = " ".join(content.get("body-text", [])) if isinstance(content.get("body-text", []), list) else str(content.get("body-text", ""))
    
    rating_data = content.get("rating", [[]])
    
    try:
        if origin == "Politifact":
            original_rating = rating_data[0][2] if len(rating_data) > 0 and len(rating_data[0]) > 2 else ""
            original_rating = re.sub(r"[^a-zA-Z0-9\s]", " ", str(original_rating)).strip()
        elif origin == "Snopes":
            original_rating = rating_data[0].split('/')[-1].replace('.png', '').strip() if len(rating_data) > 0 else ""
        else:
            raise ValueError(f"Unknown origin: {origin}")
    except Exception as e:
        print(f"Error extracting rating: {e}")
        original_rating = ""
    
    normalized_rating = normalize_rating(original_rating)
    binary_label, multiclass_label = rating_to_numeric(normalized_rating)
    
    clean_body = clean_text(body_text)
    
    return {
        "claim": claim,
        "body_text": clean_body,
        "binary_label": binary_label.value,
        "multiclass_label": multiclass_label.value,
        "rating": normalized_rating.value,
        "original_rating": original_rating
    }


def preprocess_articles(input_file, output_file, origin):
    """Preprocess articles from a JSON file and save to CSV"""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    rows = []
    for url, content in data.items():
        try:
            rows.append(extract_content(content, origin))
        except Exception as e:
            print(f"Error processing {url}: {e}")
    
    df = pd.DataFrame(rows)
    
    if len(df) == 0:
        print("No data to process!")
        return
    
    try:
        train, temp = train_test_split(
            df, 
            test_size=0.3, 
            random_state=42, 
            stratify=df['multiclass_label']
        )
        val, test = train_test_split(
            temp, 
            test_size=0.5, 
            random_state=42, 
            stratify=temp['multiclass_label']
        )
    except Exception as e:
        print(f"Error in train-test split: {e}")
        train, temp = train_test_split(df, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    train.to_csv(output_file.replace(".csv", "_train.csv"), index=False)
    val.to_csv(output_file.replace(".csv", "_val.csv"), index=False)
    test.to_csv(output_file.replace(".csv", "_test.csv"), index=False)
    
    print(f"Data saved with shapes: Train {train.shape}, Val {val.shape}, Test {test.shape}")

