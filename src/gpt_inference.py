import openai

def gpt_inference(texts, api_key):
    """
    Performs inference with GPT on the provided texts.

    Args:
        texts (list): List of texts for classification.
        api_key (str): API key to access OpenAI GPT.

    Returns:
        list: List of predictions (1 for true, 0 for false).
    """
    openai.api_key = api_key
    results = []

    for text in texts:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Text: {text}\nIs the news true or false? Respond 'True' or 'False'.",
            max_tokens=10
        )
        prediction = 1 if "True" in response.choices[0].text else 0
        results.append(prediction)
    
    return results