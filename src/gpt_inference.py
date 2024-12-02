import os
import hashlib
import json
import time
from openai import OpenAI

def gpt_inference(texts, cache_dir="cache"):
    """
    Performs inference with GPT on the provided texts, using cache to store responses.

    Args:
        texts (list): List of texts for classification.
        api_key (str): API key to access OpenAI GPT.
        cache_dir (str): Directory to store cached responses (defaults to 'cache').

    Returns:
        list: List of predictions (1 for true, 0 for false).
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    client = OpenAI(api_key= os.getenv('GPT_API_KEY'))
    results = []

    for text in texts:
        cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
        cache_file = os.path.join(cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_file):
            print(f"Using cached result for: {text}")
            with open(cache_file, "r") as f:
                cached_response = json.load(f)
            results.append(cached_response['prediction'])
        else:
            try:
                response = client.chat.completions.create(
                    messages=[{
                        "role": "user",
                        "content": f"Text: {text}\nIs the news true or false? Respond 'True' or 'False'."
                    }],
                    model="gpt-4o",
                )

                prediction = 1 if "True" in response.choices[0].message.content else 0
                results.append(prediction)

                with open(cache_file, "w") as f:
                    json.dump({"text": text, "prediction": prediction}, f)

                time.sleep(10)

            except Exception as e:
                print(f"Error occurred for text: {text}. Error: {e}")
                results.append(None)

    return results
