from sklearn.metrics import classification_report

def evaluate_model(predictions, labels):
    """
    Evaluates the model based on the predictions and the true labels.

    Args:
        predictions (list): List of model predictions.
        labels (list): List of true labels.
    """
    print(classification_report(labels, predictions))
