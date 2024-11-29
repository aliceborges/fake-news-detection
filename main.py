import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from src.data_preprocessing import preprocess_articles
from src.model_training import train_bert_model
from src.gpt_inference import gpt_inference
from src.utils import load_data, save_results
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer
import torch
from sklearn.utils.class_weight import compute_class_weight


load_dotenv()
# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def evaluate_model(texts, true_labels, model, output_file):
    """
    Evaluates the model on the provided texts and saves the results.

    Args:
        texts (list): List of texts for evaluation.
        true_labels (list): List of true labels for the texts.
        model: The trained model used for making predictions.
        output_file (str): Path to save the evaluation results.
    """

    texts =  [str(text) for text in texts]

    # Tokenize the texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Make predictions using the model
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)
        logits = outputs.logits  # Get the logits from the model output

    # Convert logits to predicted labels
    predictions = torch.argmax(logits, dim=1).numpy()  # Get the predicted class indices

    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    # Create a DataFrame to save the results
    results = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Score': [accuracy, precision, recall, f1]
    }
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    results_df.to_csv(output_file, index=False)
    print(f"Evaluation results saved to {output_file}")



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
    # Load training data
    train_data = load_data(train_data_file)
    train_texts, train_labels = train_data['body_text'], train_data['label']

    # Train BERT model
    print(f"Starting BERT training with {len(train_texts)} samples...")
    trained_model = train_bert_model(train_texts, train_labels)

    # Load validation data for evaluation
    val_data = load_data(val_data_file)
    val_texts, val_labels = val_data['body_text'], val_data['label']
    
    # Evaluate the model on validation data
    print(f"Evaluating on validation data with {len(val_texts)} samples...")
    evaluate_model(val_texts, val_labels, trained_model, './results/bert_evaluation_results.csv')

    '''
    # Load test data for inference
    test_data = load_data(test_data_file)
    test_texts = test_data['body_text']

    # Inference with GPT
    print(f"Starting GPT inference on test data...")
    gpt_api_key = os.getenv('GPT_API_KEY')
    gpt_results = gpt_inference(test_texts, api_key=gpt_api_key)
    
    # Save results
    save_results(gpt_results, gpt_results_file)
    print(f"GPT results saved to {gpt_results_file}")
    '''


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
    # Define file paths
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

    # Preprocess and save raw data
    preprocess_and_save(source_dest_pairs)

    # Merge and save processed datasets
    merge_and_save_datasets(train_paths, val_paths, test_paths, output_paths)

    # Train and perform inference
    train_and_infer_with_bert_and_gpt(
        output_paths[0],  # Train file
        output_paths[1],  # Validation file
        output_paths[2],  # Test file
        './results/gpt_results_politifact.csv',
    )


if __name__ == "__main__":
    main()