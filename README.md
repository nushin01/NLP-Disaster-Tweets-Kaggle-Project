# Natural Language Processing with Disaster Tweets

This project is based on the Kaggle competition **Natural Language Processing with Disaster Tweets**. The objective is to build a model that can classify whether a tweet is about a real disaster or not using NLP techniques.

## Problem Statement

The goal is to predict if a given tweet refers to a real disaster (1) or not (0).  
This is a binary text classification problem.

The idea behind this type of model is that it could help organizations or emergency response teams quickly identify important disaster-related information from social media.

---

## Dataset

The dataset is taken from Kaggle and contains the following files:

- `train.csv` – labeled training data (about 7613 tweets)
- `test.csv` – test data without labels
- `sample_submission.csv` – example format for submission

### Columns in the dataset:
- `id` – unique identifier for each tweet  
- `keyword` – keyword from the tweet (sometimes missing)  
- `location` – location mentioned in the tweet (many missing values)  
- `text` – actual tweet content  
- `target` – label (1 = disaster, 0 = not disaster)

---

## Exploratory Data Analysis (EDA)

Some key observations from the data:

- The dataset is slightly imbalanced, with more non-disaster tweets than disaster tweets
- A large number of missing values exist in the `location` column
- Tweets are generally short, mostly under 150 characters
- The `text` column is the most important feature for prediction

### Data Cleaning Steps:
- Converted all text to lowercase
- Removed URLs
- Removed special characters and punctuation
- Stripped extra spaces

---

## Feature Engineering

For text processing, the following steps were used:

- Tokenization using Keras `Tokenizer`
- Conversion of text into sequences
- Padding sequences to a fixed length (100)
- Used pre-trained **GloVe embeddings (100-dimensional)** to represent words

---

## Model Architecture

Two models were built and compared.

### 1. LSTM Model
- Embedding layer using pre-trained GloVe (frozen)
- LSTM layer with 64 units
- Dropout layer (0.5)
- Dense output layer with sigmoid activation

### 2. Bidirectional LSTM Model
- Same embedding layer as above
- Bidirectional LSTM layer with 64 units
- Dropout layer (0.5)
- Dense output layer with sigmoid activation

---

## Model Performance

| Model               | Validation Accuracy | Validation Loss |
|--------------------|---------------------|-----------------|
| LSTM               | 0.8181              | 0.4217          |
| Bidirectional LSTM | 0.8260              | 0.4245          |

The Bidirectional LSTM performed slightly better and was selected as the final model.

---

## Training Details

- Optimizer: Adam  
- Loss function: Binary Crossentropy  
- Batch size: 32  
- Epochs: 5  
- Max sequence length: 100  
- Word embeddings: GloVe (100d)

---

## Results and Observations

- Bidirectional LSTM performed better because it captures context from both directions in the sentence
- Pre-trained GloVe embeddings helped improve performance and reduced training time
- Dropout helped reduce overfitting
- The model trained fairly quickly and showed stable performance across epochs

---

## Kaggle Submission

The final model was used to generate predictions on the test dataset. The output was saved in:
