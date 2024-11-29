import openai
import time

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
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Text: {text}\nIs the news true or false? Respond 'True' or 'False'."}]
        )
        prediction = 1 if "True" in response.choices[0].message['content'] else 0  # Updated to access the new response format
        results.append(prediction)

        time.sleep(30)
    
    return results