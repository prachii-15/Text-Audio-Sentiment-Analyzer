import speech_recognition as sr
import pandas as pd
import os

# Initialize recognizer
recognizer = sr.Recognizer()

# Path to the folder where audio files are stored
audio_folder = r"E:\Elite Technocrats\Project\audios"  # Replace with your actual folder path

# Path to the CSV file where transcriptions will be saved
output_csv = "transcribed_audio.csv"

# Check if the CSV file exists
file_exists = os.path.exists(output_csv)

# Loop through all audio files in the audio folder
for file in os.listdir(audio_folder):
    if file.endswith(".wav"):  # Only process .wav files
        file_path = os.path.join(audio_folder, file)
        print(f"🔄 Processing {file_path}...")

        # Use the audio file as the source for recognition
        with sr.AudioFile(file_path) as source:
            # Adjust for ambient noise to improve accuracy
            recognizer.adjust_for_ambient_noise(source)

            # Listen to the audio file
            audio = recognizer.record(source)

            try:
                # Recognize speech using Google Web Speech API
                print("⏳ Recognizing...")
                text = recognizer.recognize_google(audio, language='en-US')  
                print(f"📝 You said: {text}")

                # Save the transcribed text in the CSV file
                data = [[text]]
                df = pd.DataFrame(data, columns=["Transcribed Text"])

                # Append the result to the CSV file
                df.to_csv(output_csv, mode='a', header=not file_exists, index=False)

                # Set file_exists to True after the first write (to avoid duplicate headers)
                file_exists = True

            except sr.UnknownValueError:
                print(f"❌ Could not understand audio in {file_path}.")
            except sr.RequestError as e:
                print(f"❌ Could not request results from Google Speech Recognition for {file_path}; {e}")

print(f"📄 All transcriptions saved to {output_csv}")
