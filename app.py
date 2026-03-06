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
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("amazon_alexa.tsv", sep='\t')

# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['cleaned_reviews'] = df['verified_reviews'].apply(clean_text)

# VADER scoring and label assignment
def assign_sentiment(score):
    if score >= 0.5:
        return 1  # Positive
    elif score <= -0.5:
        return 0  # Negative
    else:
        return 2  # Neutral

df['compound'] = df['verified_reviews'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
df['label'] = df['compound'].apply(assign_sentiment)

# Save the updated dataset
df.to_csv("amazon_alexa_with_sentiment.csv", index=False)

# TF-IDF with n-grams
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['cleaned_reviews'])
y = df['label']

# Handle imbalance using SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Model training
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

lr_model = LogisticRegression(class_weight='balanced', solver='liblinear', multi_class='ovr')
lr_model.fit(X_train, y_train)

# Save models
pickle.dump(nb_model, open("naive_bayes.pkl", "wb"))
pickle.dump(lr_model, open("logistic_regression.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Load models back
nb_model = pickle.load(open("naive_bayes.pkl", "rb"))
lr_model = pickle.load(open("logistic_regression.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

prediction_mapping = {0: "Negative", 1: "Positive", 2: "Neutral"}

# Audio transcription
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio, language='en-US')
        except:
            return "Could not transcribe"

# Audio feature extraction function
def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features = np.mean(mfccs, axis=1)
        if len(features) < 13:
            features = np.pad(features, (0, 13 - len(features)))
        else:
            features = features[:13]
        return features.reshape(1, -1)
    except Exception as e:
        print(f"⚠️ Audio Feature Extraction Error: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        review = request.form.get("review", "").strip()
        audio = request.files.get("audio_file")

        if review:
            cleaned = clean_text(review)
            vectorized = vectorizer.transform([cleaned])
            vader_score = sia.polarity_scores(review)['compound']

            # Handle very short input separately (e.g., 1 word)
            if len(cleaned.split()) <= 1:  # If it's a single word or empty
                if vader_score >= 0.5:
                    vader_label = "Positive"
                elif vader_score <= -0.5:
                    vader_label = "Negative"
                else:
                    vader_label = "Neutral"
                final = vader_label
                pred_nb = pred_lr = vader_label  # Directly using VADER for single-word input
            else:
                # Applying models only when input is not a single word
                nb_pred = nb_model.predict(vectorized)[0]
                lr_pred = lr_model.predict(vectorized)[0]
                pred_nb = prediction_mapping[nb_pred]
                pred_lr = prediction_mapping[lr_pred]

                # Voting logic
                votes = [pred_nb, pred_lr]
                if vader_score >= 0.5:
                    vader_label = "Positive"
                elif vader_score <= -0.5:
                    vader_label = "Negative"
                else:
                    vader_label = "Neutral"
                votes.append(vader_label)
                
                final = max(set(votes), key=votes.count)

            return render_template("index.html", review=review, sentiment_score=vader_score,
                                   sentiment_label=vader_label, prediction_nb=pred_nb,
                                   prediction_lr=pred_lr, final_prediction=final)

        elif audio and audio.filename != "":  # Audio handling section
            file_path = "temp_audio.wav"
            audio.save(file_path)

            try:
                # Extract audio features (if needed in the future, but not used for prediction now)
                audio_features = extract_audio_features(file_path)
                if audio_features is not None:
                    print(f"Extracted Audio Features: {audio_features}")
                
                # Transcribe the audio
                transcribed = transcribe_audio(file_path)
                cleaned = clean_text(transcribed)
                vectorized = vectorizer.transform([cleaned])
                vader_score = sia.polarity_scores(transcribed)['compound']

                # Handle short transcribed inputs
                if len(cleaned.split()) <= 1:  # Single-word transcriptions
                    if vader_score >= 0.5:
                        vader_label = "Positive"
                    elif vader_score <= -0.5:
                        vader_label = "Negative"
                    else:
                        vader_label = "Neutral"
                    final = vader_label
                    pred_nb = pred_lr = vader_label  # Directly using VADER for short transcriptions
                else:
                    # Apply ML models for longer inputs
                    nb_pred = nb_model.predict(vectorized)[0]
                    lr_pred = lr_model.predict(vectorized)[0]
                    pred_nb = prediction_mapping[nb_pred]
                    pred_lr = prediction_mapping[lr_pred]

                    # Voting logic
                    votes = [pred_nb, pred_lr]
                    if vader_score >= 0.5:
                        vader_label = "Positive"
                    elif vader_score <= -0.5:
                        vader_label = "Negative"
                    else:
                        vader_label = "Neutral"
                    votes.append(vader_label)

                    final = max(set(votes), key=votes.count)
            except Exception as e:
                transcribed = "Error processing audio"
                vader_score = pred_nb = pred_lr = final = None
                print("[Audio Error]:", e)
            finally:
                os.remove(file_path)

            return render_template("index.html", transcribed_text=transcribed, sentiment_score=vader_score,
                                   sentiment_label=vader_label, prediction_nb=pred_nb,
                                   prediction_lr=pred_lr, final_prediction=final)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
