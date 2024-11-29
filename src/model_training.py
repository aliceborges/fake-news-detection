import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch

class FakeNewsDataset(torch.utils.data.Dataset):
    """
    Custom dataset for training the BERT model.

    Args:
        encodings (dict): Tokenized data.
        labels (list): Corresponding labels.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract labels from inputs
        labels = inputs.pop("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs[0]  # Get the logits from the model output
        
        # Compute the loss using class weights
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

def train_bert_model(texts, labels, output_dir='./results', epochs=3, subset_size=None):
    """
    Trains a BERT model for fake news classification.

    Args:
        texts (list): List of processed texts for training.
        labels (list): List of labels (true/false).
        output_dir (str): Directory to save the training results.
        epochs (int): Number of training epochs.
        subset_size (int): Number of samples to be used for training.
    """
    # If subset_size is provided, use only that number of samples
    if subset_size:
        texts = texts[:subset_size]
        labels = labels[:subset_size]

    # Split the data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    train_texts = [str(text) for text in train_texts]
    val_texts = [str(text) for text in val_texts]

    # Tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

    # Create datasets
    train_dataset = FakeNewsDataset(train_encodings, list(train_labels))
    val_dataset = FakeNewsDataset(val_encodings, list(val_labels))

    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')  # Move to GPU if necessary

    # BERT model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Training configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    # Training
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        class_weights=class_weights,  # Pass class weights to the custom trainer
    )
    
    trainer.train()
    print("Training completed.")

    return model