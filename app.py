from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import threading
import speech_recognition as sr
import time
from textblob import TextBlob

app = Flask(__name__)

# Load the object detection model (using MobileNet for example)
net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt', 'models/mobilenet_iter_73000.caffemodel')

with open('labels.txt', 'r') as f:
    labels = f.read().strip().split('\n')

detected_objects = []
transcriptions = []
threat_analysis = []
lock = threading.Lock()  # Lock for thread-safe operations
lock_transcription = threading.Lock()  # Lock for transcription

threat_keywords = [
    'help', 'danger', 'risk', 'emergency', 'threat', 'attack', 'violence', 'weapon', 'gun', 'knife', 
    'hostage', 'assault', 'terror', 'panic', 'crisis', 'harm', 'murder', 'shooting', 'explosion', 'bomb', 
    'fight', 'kill', 'homicide', 'break-in', 'robbery', 'intruder', 'kidnapping', 'assassination', 
    'terrorist', 'deadly', 'death', 'hazard', 'fear', 'escape', 'fire', 'flee', 'abuse', 'trap', 'safety', 
    'survival'
]

emotion_keywords = {
    'Fear': [
        'help', 'danger', 'emergency', 'threat', 'scared', 'afraid', 'terrified', 'panicked', 
        'anxious', 'worried', 'frightened', 'horror', 'nervous', 'uneasy', 'dread', 'alarmed', 
        'shaking', 'intimidated', 'paranoid'
    ],
    'Anger': [
        'hate', 'angry', 'rage', 'furious', 'annoyed', 'irritated', 'frustrated', 'mad', 'resentment', 
        'hostile', 'wrath', 'aggressive', 'fury', 'enraged', 'bitter', 'offended', 'disgusted', 'outraged', 
        'infuriated', 'provoked'
    ],
    'Sadness': [
        'sad', 'cry', 'unhappy', 'depressed', 'mourn', 'grief', 'heartbroken', 'melancholy', 'sorrow', 
        'down', 'tearful', 'lonely', 'miserable', 'regret', 'despair', 'hopeless', 'gloomy', 'discouraged', 
        'helpless', 'disheartened'
    ],
    'Joy': [
        'happy', 'excited', 'joyful', 'celebrate', 'delighted', 'content', 'cheerful', 'elated', 
        'blissful', 'ecstatic', 'overjoyed', 'satisfied', 'thrilled', 'gleeful', 'grateful', 'enthusiastic', 
        'radiant', 'jubilant', 'optimistic', 'smiling'
    ],
    'Surprise': [
        'shocked', 'surprised', 'amazing', 'astonished', 'stunned', 'speechless', 'flabbergasted', 
        'dumbfounded', 'baffled', 'astounded', 'incredible', 'unbelievable', 'unexpected', 'wow', 
        'startled', 'taken aback'
    ],
    'Disgust': [
        'disgusting', 'gross', 'revolting', 'nauseating', 'repulsive', 'sickening', 'vile', 'abhorrent', 
        'loathsome', 'foul', 'horrible', 'offensive', 'detestable', 'putrid', 'odious', 'repugnant', 
        'disdainful', 'vomit', 'contempt'
    ]
}


# Function to generate video frames
def generate_frames():
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Error: Camera could not be opened.")
        return
    
    while True:
        success, frame = camera.read()
        if not success:
            break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)

        with lock:
            detected_objects.clear()

        detections = net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                with lock:
                    detected_objects.append({
                        "class": labels[idx],
                        "confidence": float(confidence)
                    })

                label = f"{labels[idx]}: {confidence:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Function to detect emotion using keywords
def detect_emotion(transcription):
    detected_emotion = "Neutral"

    for emotion, keywords in emotion_keywords.items():
        if any(keyword in transcription.lower() for keyword in keywords):
            detected_emotion = emotion
            break

    return detected_emotion

# Function to detect emotion using sentiment analysis
def detect_emotion_using_sentiment(transcription):
    blob = TextBlob(transcription)
    sentiment = blob.sentiment.polarity

    if sentiment < -0.5:
        return "Sadness"
    elif sentiment > 0.5:
        return "Joy"
    else:
        return "Neutral"

# Function to recognize audio and analyze threats/emotions
def recognize_audio():
    global transcriptions, threat_analysis
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    while True:
        with microphone as source:
            print("Listening for audio...")
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5)
                transcription = recognizer.recognize_google(audio)
                print(f"Transcription: {transcription}")

                person_detected = any(obj['class'] == 'person' for obj in detected_objects)
                threat_score = sum(keyword in transcription.lower() for keyword in threat_keywords)

                if person_detected:
                    if threat_score >= 1.5:
                        danger_level = "High"
                        color = "red"
                    elif threat_score > 0.5:
                        danger_level = "Moderate"
                        color = "orange"
                    else:
                        danger_level = "Low"
                        color = "green"
                else:
                    danger_level = "Low"
                    color = "green"

                # Detect emotion
                emotion_by_keyword = detect_emotion(transcription)
                emotion_by_sentiment = detect_emotion_using_sentiment(transcription)

                analysis_result = {
                    "danger_level": danger_level,
                    "emotion_keyword": emotion_by_keyword,
                    "emotion_sentiment": emotion_by_sentiment
                }

                with lock_transcription:
                    transcriptions.append(transcription)
                    threat_analysis.append(analysis_result)

            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Error: {e}")

        time.sleep(5)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detected_objects')
def get_detected_objects():
    with lock:
        return jsonify({"detected_objects": detected_objects})

@app.route('/transcriptions')
def get_transcriptions():
    with lock_transcription:
        return jsonify({
            "transcriptions": transcriptions,
            "analysis_results": threat_analysis
        })

if __name__ == '__main__':
    audio_thread = threading.Thread(target=recognize_audio)
    audio_thread.start()
    app.run(debug=True)
