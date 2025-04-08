from flask import Flask, render_template, request
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import speech_recognition as sr
import librosa
import os
from sklearn.preprocessing import StandardScaler

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load Models & Vectorizer
nb_model = pickle.load(open("naive_bayes.pkl", "rb"))
lr_model = pickle.load(open("logistic_regression.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Ensure audio-specific model is used
audio_model = pickle.load(open("audio_model.pkl", "rb"))
audio_scaler = pickle.load(open("audio_scaler.pkl", "rb"))

sia = SentimentIntensityAnalyzer()

# Text Preprocessing Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

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

def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        if len(y) == 0:
            print("⚠️ No audio data found!")
            return None
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features = np.mean(mfccs, axis=1)

        # Ensure feature vector matches expected shape
        required_features = 13  
        if len(features) < required_features:
            features = np.pad(features, (0, required_features - len(features)), mode='constant')
        else:
            features = features[:required_features]  

        return features.reshape(1, -1)
    except Exception as e:
        print(f"⚠️ Audio Feature Extraction Error: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "review" in request.form:
            review = request.form["review"]
            cleaned_review = clean_text(review)

            # VADER Sentiment Score
            sentiment_score = sia.polarity_scores(cleaned_review)['compound']
            sentiment_label = "Positive" if sentiment_score > 0.05 else "Negative" if sentiment_score < -0.05 else "Neutral"

            # Predict Sentiment using ML Models
            review_vectorized = vectorizer.transform([cleaned_review])
            prediction_nb = nb_model.predict(review_vectorized)[0]
            prediction_lr = lr_model.predict(review_vectorized)[0]

            print(f"TEXT SENTIMENT SCORE: {sentiment_score}, VADER: {sentiment_label}, LR: {prediction_lr}, NB: {prediction_nb}")  

            return render_template("index.html", review=review, sentiment_score=sentiment_score, 
                                   sentiment_label=sentiment_label, prediction_nb=prediction_nb, 
                                   prediction_lr=prediction_lr)
        
        elif "audio_file" in request.files and request.files["audio_file"].filename != "":
            audio_file = request.files["audio_file"]
            file_path = "temp_audio.wav"
            audio_file.save(file_path)

            try:
                # Transcribe Audio
                transcribed_text = transcribe_audio(file_path)

                # Extract Features for Model Prediction
                audio_features = extract_audio_features(file_path)
                if audio_features is None:
                    raise ValueError("Invalid audio file. Could not extract features.")

                # Scale and Predict Sentiment for Audio
                audio_features_scaled = audio_scaler.transform(audio_features)
                audio_prediction = audio_model.predict(audio_features_scaled)[0]  

                # Generate Sentiment Score
                sentiment_score = sia.polarity_scores(transcribed_text)['compound']
                sentiment_label = "Positive" if sentiment_score > 0.05 else "Negative" if sentiment_score < -0.05 else "Neutral"

                print(f"AUDIO SENTIMENT SCORE: {sentiment_score}, VADER: {sentiment_label}, ML Model: {audio_prediction}")  

            except Exception as e:
                transcribed_text = "Error processing audio"
                sentiment_score = None
                sentiment_label = None
                audio_prediction = None
                print(f"⚠️ Audio Processing Error: {e}")

            finally:
                os.remove(file_path)  # Remove temp audio file after processing

            return render_template("index.html", transcribed_text=transcribed_text, 
                                   sentiment_score=sentiment_score, sentiment_label=sentiment_label, 
                                   prediction=audio_prediction)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
