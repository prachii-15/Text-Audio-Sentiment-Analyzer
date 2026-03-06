import speech_recognition as sr
import pandas as pd
import os
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize recognizer
recognizer = sr.Recognizer()
analyzer = SentimentIntensityAnalyzer()

# Path to the folder where audio files are stored
audio_folder = r"E:\Elite Technocrats\Project\audios" 

# Path to the CSV file where transcriptions & sentiment analysis will be saved
output_csv = "transcribed_audio_with_sentiment.csv"

# Check if the CSV file exists
file_exists = os.path.exists(output_csv)

# List to store results
data = []

# Loop through all audio files in the audio folder
for file in os.listdir(audio_folder):
    if file.endswith(".wav"):  # Only process .wav files
        file_path = os.path.join(audio_folder, file)
        print(f"🔄 Processing {file_path}...")

        with sr.AudioFile(file_path) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.record(source)

            try:
                print("⏳ Recognizing...")
                text = recognizer.recognize_google(audio, language='en-US')
                print(f"📝 Transcribed: {text}")

                # Sentiment Analysis using TextBlob (Fallback)
                blob_sentiment = TextBlob(text).sentiment.polarity
                blob_label = "Positive" if blob_sentiment > 0 else "Negative" if blob_sentiment < 0 else "Neutral"

                # Sentiment Analysis using VADER
                vader_score = analyzer.polarity_scores(text)
                vader_label = "Positive" if vader_score['compound'] > 0 else "Negative" if vader_score['compound'] < 0 else "Neutral"

                # Append results to list
                data.append([text, blob_label, vader_label])

            except sr.UnknownValueError:
                print(f"❌ Could not understand audio in {file_path}.")
                data.append(["Could not transcribe", "N/A", "N/A"])
            except sr.RequestError as e:
                print(f"❌ API Request Error for {file_path}; {e}")
                data.append(["API Request Error", "N/A", "N/A"])

df = pd.DataFrame(data, columns=["Transcribed Text", "TextBlob Sentiment", "VADER Sentiment"])
df.to_csv(output_csv, mode='a', header=not file_exists, index=False)
print(f"📄 All transcriptions with sentiment analysis saved to {output_csv}")
