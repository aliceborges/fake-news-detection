# Fake News Detection System Using BERT and GPT

This project provides a comprehensive solution for detecting fake news by leveraging BERT for classification and GPT for generating inferences. The workflow includes preprocessing news articles, training a BERT model, and utilizing the OpenAI GPT model for inference.

## Contents
- [Installation Instructions](#installation-instructions)
- [How to Use](#how-to-use)
- [Directory Structure](#directory-structure)
- [System Requirements](#system-requirements)
- [License Information](#license-information)

## Installation Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
   ```

2. Set up a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows use `venv\Scripts\activate`
   ```

3. Install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and include your OpenAI API key:
   ```plaintext
   GPT_API_KEY='your_openai_api_key_here'
   ```

## How to Use

1. Preprocess the data and train the model:
   ```bash
   python main.py
   ```

2. The processed data will be stored in the `data/processed` directory, while the results will be saved in the `results` directory.

## Directory Structure

```
fake-news-detection/
│
├── data/                  # Contains both processed and raw data files
│   ├── processed/         # Contains processed data files
│   └── raw/               # Contains raw data files
│
├── results/               # Directory for output results
│
├── src/                   # Source code directory
│   ├── data_preprocessing.py
│   ├── gpt_inference.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
│
├── .env                   # File for environment variables
├── .gitignore             # Git ignore configuration
├── main.py                # Main execution script
└── requirements.txt       # List of Python dependencies
```


## System Requirements

- Python version 3.7 or higher
- `transformers`
- `torch`
- `pandas`
- `scikit-learn`
- `nltk`
- `python-dotenv`
- `openai`