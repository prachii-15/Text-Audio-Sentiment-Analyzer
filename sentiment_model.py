from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import librosa
import speech_recognition as sr
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import os
import re
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
df = pd.read_csv("amazon_alexa.tsv", sep='\t')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['cleaned_reviews'] = df['verified_reviews'].apply(clean_text)

sia = SentimentIntensityAnalyzer()

def assign_sentiment(score):
    if score >= 0.5:
        return 1
    elif score <= -0.5:
        return 0
    else:
        return 2

df['compound'] = df['verified_reviews'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
df['label'] = df['compound'].apply(assign_sentiment)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_reviews'])
y = df['label']

# Handle Class Imbalance
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Models
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

lr_model = LogisticRegression(class_weight='balanced', solver='liblinear', multi_class='ovr')
lr_model.fit(X_train, y_train)

# Evaluate naive bayes
print("=== Naive Bayes Evaluation ===")
print("Accuracy:", accuracy_score(y_test, nb_model.predict(X_test)))
print("Classification Report:\n", classification_report(y_test, nb_model.predict(X_test)))
#print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))

# Evaluate Logistic Regression
print("=== Logistic Regression Evaluation ===")
print("Accuracy:", accuracy_score(y_test, lr_model.predict(X_test)))
print("Classification Report:\n", classification_report(y_test, lr_model.predict(X_test)))
#print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# Save Models & Vectorizer
pickle.dump(nb_model, open("naive_bayes.pkl", "wb"))
pickle.dump(lr_model, open("logistic_regression.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Audio Processing Functions
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language='en-US')
            return text
        except sr.UnknownValueError:
            return "Could not transcribe"
        except sr.RequestError:
            return "API Request Error"

prediction_mapping = {0: "Negative", 1: "Positive", 2: "Neutral"}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        review = request.form["review"]
        cleaned_review = clean_text(review)
        review_vectorized = vectorizer.transform([cleaned_review])
        
        sentiment_score = sia.polarity_scores(review)['compound']
        if sentiment_score >= 0.5:
            sentiment_label = "Positive"
        elif sentiment_score <= -0.5:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        # ML Model Predictions
        pred_lr = lr_model.predict(review_vectorized)[0]
        pred_nb = nb_model.predict(review_vectorized)[0]

        prediction_lr = prediction_mapping.get(pred_lr, "Unknown")
        prediction_nb = prediction_mapping.get(pred_nb, "Unknown")

        return render_template("index.html", review=review, sentiment_score=sentiment_score,
                               sentiment_label=sentiment_label,
                               prediction_lr=prediction_lr, prediction_nb=prediction_nb)
    return render_template("index.html")

@app.route("/analyze-audio", methods=["POST"])
def analyze_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded"})
    audio_file = request.files["audio"]
    file_path = "temp.wav"
    audio_file.save(file_path)
    
    try:
        transcription = transcribe_audio(file_path)
        cleaned_transcription = clean_text(transcription)
        transcription_vectorized = vectorizer.transform([cleaned_transcription])

        sentiment_score = sia.polarity_scores(transcription)['compound']
        if sentiment_score >= 0.5:
            sentiment_label = "Positive"
        elif sentiment_score <= -0.5:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        pred_lr = lr_model.predict(transcription_vectorized)[0]
        pred_nb = nb_model.predict(transcription_vectorized)[0]

        prediction_lr = prediction_mapping.get(pred_lr, "Unknown")
        prediction_nb = prediction_mapping.get(pred_nb, "Unknown")

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        os.remove(file_path)

    return jsonify({
        "transcription": transcription,
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "prediction_lr": prediction_lr,
        "prediction_nb": prediction_nb
    })
if __name__ == "__main__":
    app.run(debug=True)
