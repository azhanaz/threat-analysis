import speech_recognition as sr
from textblob import TextBlob
import tkinter as tk
from tkinter import scrolledtext
import threading
import time
import subprocess
from dangerous_keywords import dangerous_keywords
from dangerous_phrases import dangerous_phrases

# Recursive function to analyze each word 
def recursive_threat_analysis(words, depth=0):
    if not words:
        return 0
    
    word = words[0].lower()
    threat_level = dangerous_keywords.get(word, 0)
    
    remaining_threat = recursive_threat_analysis(words[1:], depth + 1)
    
    total_threat = threat_level + remaining_threat
    
    # depth-based modifier
    if depth > 1:
        total_threat *= 1.1
    
    return total_threat

# Function to analyze dangerous phrases
def phrase_analysis(transcribed_text):
    danger_score = 0
    lower_text = transcribed_text.lower()

    # Check for dangerous phrases
    for phrase, weight in dangerous_phrases.items():
        if phrase in lower_text:
            danger_score += weight

    return danger_score

# Enhanced Sentiment Analysis Function
def analyze_sentiment(transcribed_text):
    blob = TextBlob(transcribed_text)
    sentiment_score = blob.sentiment.polarity  # Range is -1 (negative) to 1 (positive)

    # Determine emotion based on sentiment score
    if sentiment_score <= -0.8:
        emotion = "Extreme Anger / Despair"
        danger_modifier = 50
    elif sentiment_score <= -0.6:
        emotion = "Anger / Hate"
        danger_modifier = 40
    elif sentiment_score <= -0.4:
        emotion = "Fear / Sadness"
        danger_modifier = 30
    elif sentiment_score <= -0.2:
        emotion = "Discontent / Unease"
        danger_modifier = 20
    elif sentiment_score < 0.1:
        emotion = "Neutral / Slightly Negative"
        danger_modifier = 10
    elif sentiment_score < 0.4:
        emotion = "Content / Mildly Positive"
        danger_modifier = 5
    elif sentiment_score < 0.6:
        emotion = "Happy / Optimistic"
        danger_modifier = 5
    elif sentiment_score < 0.8:
        emotion = "Excited / Joyful"
        danger_modifier = 10
    else:
        emotion = "Euphoric / Elated"
        danger_modifier = 5
    
    # Return both the sentiment score and the identified emotion
    return danger_modifier, emotion

def analyze_danger_level(transcribed_text):
    # Split the transcription into words
    words = transcribed_text.split()

    # Call the recursive threat analysis function for keywords
    keyword_threat = recursive_threat_analysis(words)
    
    # Perform phrase analysis
    phrase_threat = phrase_analysis(transcribed_text)
    
    # Perform sentiment analysis and get emotion
    sentiment_threat, emotion = analyze_sentiment(transcribed_text)
    
    # Total danger level
    total_danger_level = keyword_threat + phrase_threat + sentiment_threat
    
    # Cap the total danger level at 100%
    final_danger_level = min(total_danger_level, 100)

    # Print the detected emotion
    print(f"Detected Emotion: {emotion}")

    # If danger level exceeds 55%, run the image recognition script
    if final_danger_level >= 55:
        subprocess.Popen(["python", "image_recognition.py"])
    
    return final_danger_level,emotion
# Function to transcribe audio
def transcribe_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        while True:
            try:
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                return ""

# Function to update the GUI with live transcription and analysis
def update_gui():
    while True:
        transcribed_text = transcribe_audio()
        if transcribed_text:
            danger_level, emotion = analyze_danger_level(transcribed_text)
            text_area.config(state=tk.NORMAL)
            text_area.delete(1.0, tk.END)  # Clear previous text
            text_area.insert(tk.END, f"Transcription: {transcribed_text}\n")
            text_area.insert(tk.END, f"Detected Emotion: {emotion}\n")
            text_area.insert(tk.END, f"Danger Level: {danger_level}%\n")
            text_area.config(state=tk.DISABLED)
        time.sleep(1)  # Update every second

# Create GUI window
window = tk.Tk()
window.title("Voice Analysis")

# Create a text area for displaying results
text_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=50, height=20, state=tk.DISABLED)
text_area.pack(padx=10, pady=10)

# Start the GUI update in a separate thread
thread = threading.Thread(target=update_gui)
thread.daemon = True
thread.start()

# Run the GUI main loop
window.mainloop()
