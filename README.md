# Text & Audio Sentiment Analyzer

This project performs **sentiment analysis on both text and audio inputs** using Natural Language Processing and Machine Learning techniques. The system analyzes customer reviews from the **Amazon Alexa dataset** and classifies sentiment into **Positive, Negative, or Neutral**.

For audio input, the system converts speech into text using **Speech Recognition**, extracts **MFCC (Mel Frequency Cepstral Coefficients)** features from audio signals, and then predicts sentiment using trained machine learning models.

The project demonstrates an **end-to-end sentiment analysis pipeline** including preprocessing, feature extraction, model training, and deployment using a **Flask web application**.

---

# Dataset

The project uses the **Amazon Alexa Reviews Dataset (`amazon_alexa.tsv`)**, which contains user reviews for Amazon Alexa products.

Sentiment labels are generated using **VADER Sentiment Analysis**, which produces a compound score that is mapped to three classes:

- Positive  
- Negative  
- Neutral  

---

# Features

- Text sentiment analysis
- Audio sentiment analysis
- Speech-to-text transcription
- Text preprocessing using NLP techniques
- MFCC feature extraction for audio
- TF-IDF vectorization for text
- Handling class imbalance using **SMOTE**
- Machine learning based sentiment classification
- Flask based web interface

---

# Methodology

## Text Sentiment Analysis

**1. Data Collection**
- Amazon Alexa Reviews Dataset

**2. Preprocessing**
- Text cleaning
- Tokenization
- Stopwords removal

**3. Feature Extraction**
- TF-IDF Vectorization

**4. Sentiment Classification**
- Naïve Bayes  
- Logistic Regression  
- VADER Sentiment Analysis

---

## Audio Sentiment Analysis

**1. Data Collection**
- Custom audio clips generated using **gTTS**

**2. Preprocessing**
- Audio processing using **Librosa**

**3. Feature Extraction**
- **MFCC (Mel Frequency Cepstral Coefficients)** extraction using Librosa

**4. Audio Transcription**
- Convert audio to text using **SpeechRecognition**

**5. Sentiment Classification**
- Naïve Bayes  
- Logistic Regression  
- VADER Sentiment Analysis

---

# Machine Learning Models

The following machine learning models are used for sentiment classification:

- Logistic Regression
- Multinomial Naïve Bayes

These models are trained on features extracted using **TF-IDF for text** and **MFCC for audio**.

---

# Technologies Used

- Python  
- Flask  
- Scikit-learn  
- NLTK  
- VADER Sentiment Analysis  
- Librosa  
- SpeechRecognition  
- Pandas  
- NumPy  

---

# Project Workflow

1. Load Amazon Alexa review dataset  
2. Clean and preprocess review text  
3. Generate sentiment scores using **VADER**  
4. Convert text into numerical features using **TF-IDF**  
5. Extract **MFCC features from audio signals**  
6. Handle class imbalance using **SMOTE**  
7. Train Logistic Regression and Naïve Bayes models  
8. Convert audio to text using Speech Recognition  
9. Predict sentiment for both text and audio inputs  

---

## How to Run the Project

1. Clone the repository

git clone https://github.com/your-username/Text-Audio-Sentiment-Analyzer.git

2. Install dependencies

pip install -r requirements.txt

3. Run the application

python app.py

4. Open in browser

http://127.0.0.1:5000
