from sklearn.metrics import classification_report
import openai


def evaluate_model(predictions, labels):
    """
    Evaluates the model based on the predictions and the true labels.

    Args:
        predictions (list): List of model predictions.
        labels (list): List of true labels.
    """
    print(classification_report(labels, predictions))


def validate_with_gpt(texts, predictions, confidences, threshold=0.7, api_key=None):
    '''
    Uses GPT to validate model predictions on low-confidence examples.

    Args:
        texts (list): List of texts classified by the model.
        predictions (list): Model predictions for the texts.
        confidences (list): Confidence scores for the predictions.
        threshold (float): Confidence threshold below which GPT is used for validation.
        api_key (str): OpenAI API key.

    Returns:
        list: Validated predictions with GPT corrections.
    '''
    if not api_key:
        raise ValueError("API key for GPT is required.")

    openai.api_key = api_key
    validated_predictions = []

    for text, pred, conf in zip(texts, predictions, confidences):
        if conf < threshold:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Text: {text}\\nModel Prediction: {pred}.\\nIs this classification correct? Please validate as 'True' or 'False'."}]
            )
            validated_pred = 1 if "True" in response.choices[0].message['content'] else 0
            validated_predictions.append(validated_pred)
        else:
            validated_predictions.append(pred)

    return validated_predictions
