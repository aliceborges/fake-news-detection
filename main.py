import os
from datetime import datetime
import hashlib
import pickle
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from src.data_preprocessing import preprocess_articles
from src.gpt_inference import gpt_inference
from src.model_training import train_bert
from src.utils import load_data, save_results
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                            classification_report, precision_score, 
                            recall_score, f1_score)
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


def should_run(component: str) -> bool:
    """Check if component should run based on environment variable"""
    return os.getenv(component.upper()) == 'True'


def process_source(source_name: str, input_path: str, output_dir: str) -> list:
    """Process single data source if enabled"""
    if not should_run(source_name):
        return []
    
    output_path = f"{output_dir}/{source_name}/processed_data.csv"
    os.makedirs(f"{output_dir}/{source_name}", exist_ok=True)
    preprocess_articles(input_path, output_path, source_name)
    return [
        f"{output_dir}/{source_name}/processed_data_train.csv",
        f"{output_dir}/{source_name}/processed_data_val.csv",
        f"{output_dir}/{source_name}/processed_data_test.csv"
    ]


def merge_datasets(file_paths: list, output_file: str) -> str:
    """Merge dataset files from enabled sources"""
    if not file_paths:
        return None
    pd.concat([pd.read_csv(f) for f in file_paths]).to_csv(output_file, index=False)
    return output_file


def run_bert_training(train_file: str, val_file: str, test_file: str):
    """Run BERT training and evaluation if enabled"""
    if not should_run('BERT'):
        return

    train_data = load_data(train_file)
    val_data = load_data(val_file)
    test_data = load_data(test_file)

    print(f"Starting BERT training with {len(train_data['body_text'])} samples...")
    trained_model = train_bert(
        train_data['body_text'], 
        train_data['binary_label'],
        val_data['body_text'],
        val_data['binary_label']
    )

    print(f"Testing BERT model with {len(test_data['body_text'])} samples...")
    test_model(
        test_data['body_text'], 
        test_data['binary_label'],
        trained_model, 
        './results/bert_evaluation_results.csv'
    )


def run_gpt_inference(test_file: str):
    """Run GPT inference if enabled"""
    if not should_run('GPT'):
        return

    test_data = load_data(test_file)
    print(f"Starting GPT inference on {len(test_data['body_text'])} samples...")
    gpt_results = gpt_inference(test_data['body_text'])

    gpt_results = np.asarray(gpt_results, dtype=np.int64)
    test_labels = np.asarray(test_data['binary_label'], dtype=np.int64)
    
    accuracy = accuracy_score(test_labels, gpt_results)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"./results/gpt_evaluation_results_{timestamp}.csv"
    
    pd.DataFrame({'Metric': ['Accuracy'], 'Score': [accuracy]}).to_csv(output_file, index=False)
    
    print(f"GPT evaluation results saved to {output_file}")
    print(f"GPT model accuracy: {accuracy * 100:.2f}%")


def main():
    """Main execution flow controlled by environment variables"""
    data_config = {
        'Politifact': ('data/raw/Politifact/articles_content.json', 'data/processed'),
        'Snopes': ('data/raw/Snopes/articles_content.json', 'data/processed')
    }

    processed_files = []
    for source, (input_path, output_dir) in data_config.items():
        processed_files.extend(process_source(source, input_path, output_dir))

    if not processed_files and (should_run('BERT') or should_run('GPT')):
        print("No data sources enabled but models requested. Check your .env file.")
        return

    train_files = [f for f in processed_files if 'train' in f]
    val_files = [f for f in processed_files if 'val' in f]
    test_files = [f for f in processed_files if 'test' in f]

    train_path = merge_datasets(train_files, './data/processed/train.csv')
    val_path = merge_datasets(val_files, './data/processed/val.csv')
    test_path = merge_datasets(test_files, './data/processed/test.csv')

    run_bert_training(train_path, val_path, test_path)
    run_gpt_inference(test_path)


if __name__ == "__main__":
    main()