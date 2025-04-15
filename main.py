from datetime import datetime
import hashlib
import os
import pickle
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from src.data_preprocessing import preprocess_articles
from src.gpt_inference import gpt_inference
from src.model_training import train_bert
from src.utils import load_data, save_results
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, precision_score, recall_score, f1_score
from transformers import RobertaTokenizer
import torch


load_dotenv()
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')


def test_model(texts, true_labels, model, output_file, cache_dir="cache", threshold=0.5):
    """
    Test the model on the provided texts and saves the results.
    Uses a cache file to avoid recomputing predictions for the same texts.

    Args:
        texts (list): List of texts for evaluation.
        true_labels (list): List of true labels for the texts.
        model: The trained model used for making predictions.
        output_file (str): Path to save the evaluation results.
        cache_dir (str): Directory where the cache files are stored (defaults to 'cache').
        threshold (float): Threshold for classification decision (default 0.5).
    """
    if len(texts) != len(true_labels):
        raise ValueError("The number of texts must match the number of labels.")

    cache_key = hashlib.md5(" ".join(texts).encode('utf-8')).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")

    if os.path.exists(cache_file):
        print("Using cached predictions from file.")
        with open(cache_file, "rb") as f:
            predictions = pickle.load(f)
    else:
        print("Calculating predictions...")
        texts = [str(text) for text in texts]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        model.eval()

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        probabilities = torch.softmax(logits, dim=1)
        predictions = (probabilities[:, 1] > threshold).long().numpy()

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        with open(cache_file, "wb") as f:
            pickle.dump(predictions, f)

    true_labels = np.asarray(true_labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)

    print(f"True Labels Distribution: {np.bincount(true_labels)}")
    print(f"Predicted Labels Distribution: {np.bincount(predictions)}")

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=1)
    recall = recall_score(true_labels, predictions, zero_division=1)
    f1 = f1_score(true_labels, predictions, zero_division=1)
    balanced_accuracy = balanced_accuracy_score(true_labels, predictions)

    class_report = classification_report(true_labels, predictions, zero_division=1)
    print("\nClassification Report:\n", class_report)

    results = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Balanced Accuracy'],
        'Score': [accuracy, precision, recall, f1, balanced_accuracy]
    }
    results_df = pd.DataFrame(results)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"{output_file.replace('.csv', '')}_{timestamp}.csv"
    
    results_df.to_csv(output_file, index=False)
    with open(output_file.replace(".csv", "_detailed.txt"), "w") as f:
        f.write(class_report)

    print(f"Test results saved to {output_file}")


def process_and_save_data(input_file, output_file):
    """Preprocess and save data from a JSON file."""
    preprocess_articles(input_file, output_file)
    print(f"Data processed and saved to {output_file}")


def train_and_infer_with_bert_and_gpt(train_data_file, val_data_file, test_data_file, gpt_results_file):
    """Train the BERT model and perform inference with GPT.

    Args:
        train_data_file (str): Path to the training data.
        val_data_file (str): Path to the validation data.
        test_data_file (str): Path to the test data.
        gpt_results_file (str): Path to save GPT results.
    """
    # Texts for tests
    test_data = load_data(test_data_file)
    test_texts, test_labels = test_data['body_text'], test_data['label']
    # Texts for training
    train_data = load_data(train_data_file)
    train_texts, train_labels = train_data['body_text'], train_data['label']
    # Texts for evaluation
    val_data = load_data(val_data_file)
    val_texts, val_labels = val_data['body_text'], val_data['label']

    # BERT
    print(f"Starting BERT training with {len(train_texts)} samples...")
    trained_model = train_bert(train_texts, train_labels, val_texts, val_labels)

    print(f"Test with tests data with {len(test_texts)} samples...")
    test_model(test_texts, test_labels, trained_model, './results/bert_evaluation_results.csv')
    
    #GPT
    # Inference with GPT
    print(f"Starting GPT inference on test data...")
    gpt_results = gpt_inference(test_texts)

    gpt_results = np.asarray(gpt_results, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    accuracy = accuracy_score(test_labels, gpt_results)

    print(f"GPT model accuracy: {accuracy * 100:.2f}%")
    save_results(f"GPT model accuracy: {accuracy * 100:.2f}%", gpt_results_file)


def preprocess_and_save(source_dest_pairs):
    """Preprocess and save data for multiple sources."""
    for source, destination in source_dest_pairs:
        process_and_save_data(source, destination)


def merge_and_save_datasets(train_paths, val_paths, test_paths, output_paths):
    """Merge train, validation, and test datasets and save to output paths."""
    for paths, output_file in zip([train_paths, val_paths, test_paths], output_paths):
        merged_data = pd.concat([load_data(path) for path in paths])
        merged_data.to_csv(output_file, index=False)


def main():
    """Main function to preprocess data, train, and perform inference."""
    
    source_dest_pairs = [
        ("data/raw/Politifact/articles_content.json", "data/processed/Politifact/processed_data.csv"),
        ("data/raw/Snopes/articles_content.json", "data/processed/Snopes/processed_data.csv"),
    ]

    train_paths = [
        "data/processed/Politifact/processed_data_train.csv",
        "data/processed/Snopes/processed_data_train.csv",
    ]
    val_paths = [
        "data/processed/Politifact/processed_data_val.csv",
        "data/processed/Snopes/processed_data_val.csv",
    ]
    test_paths = [
        "data/processed/Politifact/processed_data_test.csv",
        "data/processed/Snopes/processed_data_test.csv",
    ]

    output_paths = [
        './data/processed/train.csv',
        './data/processed/val.csv',
        './data/processed/test.csv',
    ]

    if not os.path.exists('data/processed/Politifact'):
        os.makedirs('data/processed/Politifact')
    if not os.path.exists('data/processed/Snopes'):
        os.makedirs('data/processed/Snopes')

    preprocess_and_save(source_dest_pairs)
    merge_and_save_datasets(train_paths, val_paths, test_paths, output_paths)
    train_and_infer_with_bert_and_gpt(
        output_paths[0],  # Train file
        output_paths[1],  # Validation file
        output_paths[2],  # Test file
        './results/gpt_results_politifact.csv',
    )


if __name__ == "__main__":
    main()