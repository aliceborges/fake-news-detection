import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.utils.class_weight import compute_class_weight
import torch
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {
        "accuracy": acc,
        "f1": f1
    }

class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = [int(label) for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def train_bert(train_texts, train_labels, val_texts, val_labels, output_dir='./results', epochs=3):
    """
    Train a BERT model for binary text classification with provided training and validation data.

    This function fine-tunes a pre-trained RoBERTa model (from the Transformers library) 
    for a binary classification task. It uses provided training and validation datasets 
    and outputs the trained model.

    Args:
        train_texts (list[str]): List of texts for training.
        train_labels (list[int]): Corresponding labels for training texts.
        val_texts (list[str]): List of texts for validation.
        val_labels (list[int]): Corresponding labels for validation texts.
        output_dir (str, optional): Directory to save model checkpoints and logs. Default is './results'.
        epochs (int, optional): Number of training epochs. Default is 3.

    Returns:
        model (BertForSequenceClassification): The fine-tuned BERT model.
    """
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    train_texts, val_texts = [str(text) for text in train_texts], [str(text) for text in val_texts]
    train_labels, val_labels = [str(text) for text in train_labels], [str(text) for text in val_labels]
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

    train_dataset = FakeNewsDataset(train_encodings, train_labels)
    val_dataset = FakeNewsDataset(val_encodings, val_labels)

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')

    model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=2)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print("Validation results:", eval_results)

    return model
