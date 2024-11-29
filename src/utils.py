import pandas as pd

def load_data(filepath):
    """
    Loads a CSV file into a DataFrame.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        DataFrame: Loaded data.
    """
    return pd.read_csv(filepath)

def save_results(predictions, filepath):
    """
    Saves the predictions to a CSV file.

    Args:
        predictions (list): List of predictions.
        filepath (str): Path to save the file.
    """
    pd.DataFrame(predictions, columns=['prediction']).to_csv(filepath, index=False)
