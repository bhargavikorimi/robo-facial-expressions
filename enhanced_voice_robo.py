import tkinter as tk
from tkinter import ttk
import speech_recognition as sr
import pyaudio
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import threading
import time
import queue
from math import cos, sin, pi
import torch
import torchaudio
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings("ignore")


class EnhancedVoiceEmotionDetector:
    """Advanced emotion detection using audio + NLP models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Load pre-trained emotion models
        try:
            # Emotion classification model
            self.emotion_classifier = pipeline(
                "text-classification", 
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            print("✓ Loaded emotion classification model")
        except Exception as e:
            print(f"Could not load emotion model: {e}")
            self.emotion_classifier = None
        
        try:
            # Sentiment analysis model
            self.sentiment_classifier = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            print("✓ Loaded sentiment analysis model")
        except Exception as e:
            print(f"Could not load sentiment model: {e}")
            self.sentiment_classifier = None
        
        # Enhanced emotion mapping
        self.emotion_keywords = {
            'angry': ['angry', 'mad', 'furious', 'rage', 'hate', 'annoyed', 'frustrated', 'upset'],
            'sad': ['sad', 'depressed', 'down', 'unhappy', 'miserable', 'gloomy', 'blue', 'melancholy'],
            'happy': ['happy', 'joy', 'excited', 'cheerful', 'glad', 'delighted', 'ecstatic', 'thrilled'],
            'fear': ['scared', 'afraid', 'terrified', 'fearful', 'anxious', 'worried', 'nervous'],
            'surprise': ['surprised', 'amazed', 'shocked', 'astonished', 'wow', 'unexpected'],
            'disgust': ['disgusted', 'revolted', 'sick', 'nauseated', 'repulsed']
        }
    
    def extract_audio_features(self, audio_data, sample_rate):
        """Extract comprehensive audio features"""
        try:
            # Convert to float32 for librosa - handle different data types
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
            
            # Extract multiple audio features
            features = {}
            
            # Fundamental frequency (pitch)
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
            pitch_values = pitches[pitches > 0]
            features['pitch_mean'] = np.mean(pitch_values) if len(pitch_values) > 0 else 150
            features['pitch_std'] = np.std(pitch_values) if len(pitch_values) > 0 else 20
            features['pitch_max'] = np.max(pitch_values) if len(pitch_values) > 0 else 200
            
            # Energy and loudness
            energy = librosa.feature.rms(y=audio_data)[0]
            features['energy_mean'] = np.mean(energy)
            features['energy_std'] = np.std(energy)
            features['energy_max'] = np.max(energy)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            
            # Zero crossing rate (speech activity)
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features['zcr_mean'] = np.mean(zcr)
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            features['tempo'] = tempo
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
            
            return features
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return {
                'pitch_mean': 150, 'energy_mean': 0.25, 'tempo': 1.5,
                'pitch_std': 20, 'energy_std': 0.1, 'pitch_max': 200,
                'energy_max': 0.5, 'spectral_centroid_mean': 2000,
                'zcr_mean': 0.1, 'mfcc_mean': [0] * 13
            }
    
    def analyze_text_emotion(self, text):
        """Analyze emotion from transcribed text using multiple NLP models"""
        if not text or len(text.strip()) < 3:
            return {'emotion': 'neutral', 'confidence': 0.5}
        
        emotion_scores = {}
        
        # Method 1: Pre-trained emotion classification
        if self.emotion_classifier:
            try:
                results = self.emotion_classifier(text)
                for result in results:
                    emotion_scores[result['label']] = result['score']
            except Exception as e:
                print(f"Emotion classification error: {e}")
        
        # Method 2: VADER sentiment analysis
        vader_scores = self.vader_analyzer.polarity_scores(text)
        if vader_scores['compound'] > 0.1:
            emotion_scores['positive'] = vader_scores['compound']
        elif vader_scores['compound'] < -0.1:
            emotion_scores['negative'] = abs(vader_scores['compound'])
        else:
            emotion_scores['neutral'] = 0.5
        
        # Method 3: TextBlob sentiment
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0.1:
                emotion_scores['positive_textblob'] = polarity
            elif polarity < -0.1:
                emotion_scores['negative_textblob'] = abs(polarity)
        except Exception as e:
            print(f"TextBlob error: {e}")
        
        # Method 4: Keyword matching
        text_lower = text.lower()
        for emotion, keywords in self.emotion_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                emotion_scores[f'{emotion}_keyword'] = min(matches / len(keywords), 1.0)
        
        # Combine and normalize scores
        final_emotion = 'neutral'
        max_confidence = 0.5
        
        # Map to our emotion categories
        emotion_mapping = {
            'joy': 'happy',
            'sadness': 'sad', 
            'anger': 'angry',
            'fear': 'sad',  # Map fear to sad for robot expressions
            'surprise': 'happy',  # Map surprise to happy
            'disgust': 'angry',  # Map disgust to angry
            'positive': 'happy',
            'negative': 'sad',
            'positive_textblob': 'happy',
            'negative_textblob': 'sad',
            'angry_keyword': 'angry',
            'sad_keyword': 'sad',
            'happy_keyword': 'happy'
        }
        
        combined_scores = {'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 0}
        
        for emotion, score in emotion_scores.items():
            mapped_emotion = emotion_mapping.get(emotion, 'neutral')
            if mapped_emotion in combined_scores:
                combined_scores[mapped_emotion] += score
        
        # Find the emotion with highest score
        if combined_scores:
            final_emotion = max(combined_scores, key=combined_scores.get)
            max_confidence = combined_scores[final_emotion]
        
        return {'emotion': final_emotion, 'confidence': max_confidence}
    
    def analyze_audio_emotion(self, audio_data, sample_rate):
        """Analyze emotion from audio features"""
        features = self.extract_audio_features(audio_data, sample_rate)
        
        # Enhanced rule-based emotion detection
        pitch = features['pitch_mean']
        energy = features['energy_mean']
        tempo = features['tempo']
        pitch_variation = features['pitch_std']
        
        # More sophisticated emotion detection
        emotion_score = {'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 0}
        
        # Happy: High pitch, high energy, moderate tempo, low pitch variation
        if pitch > 180 and energy > 0.3 and 1.5 < tempo < 2.5 and pitch_variation < 30:
            emotion_score['happy'] = 0.8
        # Angry: High pitch, very high energy, fast tempo, high pitch variation
        elif pitch > 170 and energy > 0.4 and tempo > 2.0 and pitch_variation > 25:
            emotion_score['angry'] = 0.8
        # Sad: Low pitch, low energy, slow tempo
        elif pitch < 140 and energy < 0.25 and tempo < 1.4:
            emotion_score['sad'] = 0.8
        # Neutral: Medium values
        else:
            emotion_score['neutral'] = 0.6
        
        final_emotion = max(emotion_score, key=emotion_score.get)
        confidence = emotion_score[final_emotion]
        
        return {'emotion': final_emotion, 'confidence': confidence}
    
    def detect_emotion_hybrid(self, audio_data, sample_rate, text=""):
        """Hybrid emotion detection combining audio and text analysis"""
        
        # Analyze audio features
        audio_result = self.analyze_audio_emotion(audio_data, sample_rate)
        
        # Analyze text if available
        text_result = {'emotion': 'neutral', 'confidence': 0.5}
        if text and len(text.strip()) > 2:
            text_result = self.analyze_text_emotion(text)
        
        # Combine results with weights
        audio_weight = 0.6  # Audio is more reliable for tone
        text_weight = 0.4   # Text provides semantic meaning
        
        combined_scores = {'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 0}
        
        # Add audio scores
        if audio_result['emotion'] in combined_scores:
            combined_scores[audio_result['emotion']] += audio_result['confidence'] * audio_weight
        
        # Add text scores
        if text_result['emotion'] in combined_scores:
            combined_scores[text_result['emotion']] += text_result['confidence'] * text_weight
        
        # Find final emotion
        final_emotion = max(combined_scores, key=combined_scores.get)
        final_confidence = combined_scores[final_emotion]
        
        # Confidence threshold - if too low, default to neutral
        if final_confidence < 0.3:
            final_emotion = 'neutral'
            final_confidence = 0.5
        
        return {
            'emotion': final_emotion, 
            'confidence': final_confidence,
            'audio_emotion': audio_result['emotion'],
            'text_emotion': text_result['emotion']
        }


class EnhancedVoiceControlledFaceDisplay:
    """Enhanced voice-controlled robot face with advanced emotion detection"""
    
    # Colors and sizes
    BG = "#0f1220"
    NEON = "#5ce1ff"
    TEAR = "#6fd7ff"
    BLUSH = "#ff99cc"
    
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Voice-Controlled Robo Face")
        self.root.configure(bg=self.BG)
        
        self.canvas_width = 820
        self.canvas_height = 540
        self.stroke = 16
        
        # Enhanced voice control setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.emotion_detector = EnhancedVoiceEmotionDetector()
        self.is_listening = False
        self.audio_queue = queue.Queue()
        
        # Calibrate microphone
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        
        self.setup_gui()
        self.setup_voice_control()
        
        # Initial expression
        self.draw_happy()
    
    def setup_gui(self):
        """Setup the enhanced GUI components"""
        # Main canvas
        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg=self.BG,
            highlightthickness=0,
        )
        self.canvas.pack(pady=20)
        
        # Control panel
        control_frame = tk.Frame(self.root, bg=self.BG)
        control_frame.pack(pady=10)
        
        # Voice control button
        self.voice_button = tk.Button(
            control_frame,
            text="Start Enhanced Voice Control",
            command=self.toggle_voice_control,
            bg="#162036",
            fg=self.NEON,
            activebackground="#1e2a46",
            activeforeground=self.NEON,
            relief=tk.FLAT,
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10
        )
        self.voice_button.pack(side=tk.LEFT, padx=10)
        
        # Enhanced status label
        self.status_label = tk.Label(
            control_frame,
            text="Enhanced Voice Control: OFF",
            bg=self.BG,
            fg=self.NEON,
            font=("Arial", 10)
        )
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Confidence display
        self.confidence_label = tk.Label(
            control_frame,
            text="Confidence: --",
            bg=self.BG,
            fg=self.NEON,
            font=("Arial", 10)
        )
        self.confidence_label.pack(side=tk.LEFT, padx=20)
        
        # Manual control buttons
        manual_frame = tk.Frame(self.root, bg=self.BG)
        manual_frame.pack(pady=10)
        
        btn_specs = [
            ("1\nHappy", self.draw_happy),
            ("2\nSad", self.draw_sad),
            ("3\nNeutral", self.draw_neutral),
            ("4\nLovely", self.draw_lovely),
            ("5\nLaugh", self.draw_laughing),
            ("6\nBlush", self.draw_blushing),
            ("7\nSurprise", self.draw_surprised),
            ("8\nAngry", self.draw_angry),
            ("9\nCry", self.draw_cry),
            ("0\nSleep", self.draw_sleepy),
        ]
        
        for label, handler in btn_specs:
            b = tk.Button(
                manual_frame,
                text=label,
                width=8,
                bg="#162036",
                fg=self.NEON,
                activebackground="#1e2a46",
                activeforeground=self.NEON,
                relief=tk.FLAT,
                command=handler,
            )
            b.pack(side=tk.LEFT, padx=4)
        
        # Key bindings
        self.root.bind("1", lambda _e: self.draw_happy())
        self.root.bind("2", lambda _e: self.draw_sad())
        self.root.bind("3", lambda _e: self.draw_neutral())
        self.root.bind("4", lambda _e: self.draw_lovely())
        self.root.bind("5", lambda _e: self.draw_laughing())
        self.root.bind("6", lambda _e: self.draw_blushing())
        self.root.bind("7", lambda _e: self.draw_surprised())
        self.root.bind("8", lambda _e: self.draw_angry())
        self.root.bind("9", lambda _e: self.draw_cry())
        self.root.bind("0", lambda _e: self.draw_sleepy())
    
    def setup_voice_control(self):
        """Setup enhanced voice recognition with text transcription"""
        def listen_continuously():
            while True:
                if self.is_listening:
                    try:
                        with self.microphone as source:
                            # Listen for audio
                            audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=4)
                            
                            # Get raw audio data for emotion detection
                            audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
                            sample_rate = audio.sample_rate
                            
                            # Try to transcribe speech
                            text = ""
                            try:
                                text = self.recognizer.recognize_google(audio)
                                print(f"Transcribed: {text}")
                            except sr.UnknownValueError:
                                print("Could not understand audio")
                            except sr.RequestError as e:
                                print(f"Speech recognition error: {e}")
                            
                            # Detect emotion using hybrid method
                            emotion_result = self.emotion_detector.detect_emotion_hybrid(
                                audio_data, sample_rate, text
                            )
                            
                            # Update expression based on emotion
                            self.root.after(0, lambda: self.update_expression_from_emotion(emotion_result))
                            
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as e:
                        print(f"Voice recognition error: {e}")
                        continue
                else:
                    time.sleep(0.1)
        
        # Start listening thread
        self.listen_thread = threading.Thread(target=listen_continuously, daemon=True)
        self.listen_thread.start()
    
    def toggle_voice_control(self):
        """Toggle voice control on/off"""
        self.is_listening = not self.is_listening
        if self.is_listening:
            self.voice_button.config(text="Stop Enhanced Voice Control")
            self.status_label.config(text="Enhanced Voice Control: ON")
            self.confidence_label.config(text="Confidence: --")
        else:
            self.voice_button.config(text="Start Enhanced Voice Control")
            self.status_label.config(text="Enhanced Voice Control: OFF")
            self.confidence_label.config(text="Confidence: --")
    
    def update_expression_from_emotion(self, emotion_result):
        """Update expression based on detected emotion with confidence"""
        emotion = emotion_result['emotion']
        confidence = emotion_result['confidence']
        audio_emotion = emotion_result.get('audio_emotion', 'unknown')
        text_emotion = emotion_result.get('text_emotion', 'unknown')
        
        emotion_map = {
            'angry': self.draw_angry,
            'happy': self.draw_happy,
            'sad': self.draw_sad,
            'neutral': self.draw_neutral
        }
        
        if emotion in emotion_map:
            emotion_map[emotion]()
            
            # Update status with detailed information
            status_text = f"Enhanced Voice Control: ON - Detected: {emotion.upper()}"
            if audio_emotion != 'unknown' or text_emotion != 'unknown':
                status_text += f" (Audio: {audio_emotion}, Text: {text_emotion})"
            
            self.status_label.config(text=status_text)
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}")
    
    # --------------------- Drawing helpers (same as before) ---------------------
    def clear(self):
        self.canvas.delete("all")
    
    def neon_oval(self, x0, y0, x1, y1):
        self.canvas.create_oval(
            x0, y0, x1, y1,
            outline=self.NEON,
            width=self.stroke,
        )
    
    def neon_arc(self, x0, y0, x1, y1, start, extent):
        self.canvas.create_arc(
            x0, y0, x1, y1,
            start=start,
            extent=extent,
            style=tk.ARC,
            outline=self.NEON,
            width=self.stroke,
        )
    
    def neon_line(self, x0, y0, x1, y1):
        self.canvas.create_line(
            x0, y0, x1, y1,
            fill=self.NEON,
            width=self.stroke,
            capstyle=tk.ROUND,
        )
    
    def heart(self, cx, cy, size):
        # Draw a heart using two arcs and a V line
        r = size // 2
        self.neon_arc(cx - size, cy - size, cx, cy, start=45, extent=225)
        self.neon_arc(cx, cy - size, cx + size, cy, start=-45, extent=225)
        self.neon_line(cx - r, cy, cx, cy + r)
        self.neon_line(cx + r, cy, cx, cy + r)
    
    # --------------------- Face base ---------------------
    def draw_face_base(self):
        # Eye positions
        self.left_eye = (230, 170, 350, 290)
        self.right_eye = (470, 170, 590, 290)
    
    def eyes_open(self):
        lx0, ly0, lx1, ly1 = self.left_eye
        rx0, ry0, rx1, ry1 = self.right_eye
        self.neon_oval(lx0, ly0, lx1, ly1)
        self.neon_oval(rx0, ry0, rx1, ry1)
    
    def eyes_closed(self):
        # draw two flat lines as closed eyes
        lx0, ly0, lx1, _ = self.left_eye
        rx0, ry0, rx1, _ = self.right_eye
        y = (ly0 + 40)
        self.neon_line(lx0 + 20, y, lx1 - 20, y)
        self.neon_line(rx0 + 20, y, rx1 - 20, y)
    
    def mouth_arc(self, smile):
        # smile >0 for up-curve, <0 for down-curve, 0 for neutral
        x0, y0, x1, y1 = 220, 340, 600, 520
        if smile > 0:
            self.neon_arc(x0, y0, x1, y1, start=200, extent=140)
        elif smile < 0:
            self.neon_arc(x0, y0 + 40, x1, y1 + 40, start=20, extent=140)
        else:
            self.neon_line(270, 430, 550, 430)
    
    # --------------------- Expressions (same as before) ---------------------
    def draw_happy(self):
        self.clear()
        self.draw_face_base()
        self.eyes_open()
        self.mouth_arc(smile=1)
    
    def draw_sad(self):
        self.clear()
        self.draw_face_base()
        self.eyes_open()
        self.mouth_arc(smile=-1)
    
    def draw_neutral(self):
        self.clear()
        self.draw_face_base()
        self.eyes_open()
        self.mouth_arc(smile=0)
    
    def draw_lovely(self):
        self.clear()
        self.draw_face_base()
        # Heart eyes
        self.heart(290, 230, 70)
        self.heart(530, 230, 70)
        self.mouth_arc(smile=1)
    
    def draw_laughing(self):
        self.clear()
        self.draw_face_base()
        self.eyes_closed()
        # Big open laughing mouth with slight glow effect via two strokes
        x0, y0, x1, y1 = 250, 370, 570, 520
        self.canvas.create_arc(
            x0, y0, x1, y1,
            start=200,
            extent=140,
            style=tk.ARC,
            outline=self.NEON,
            width=self.stroke + 6,
        )
        self.neon_arc(x0, y0, x1, y1, start=200, extent=140)
        # Tears of joy (small tilted droplets)
        self.canvas.create_oval(200, 320, 230, 360, outline=self.TEAR, width=self.stroke)
        self.canvas.create_oval(590, 320, 620, 360, outline=self.TEAR, width=self.stroke)
    
    def draw_blushing(self):
        self.clear()
        self.draw_face_base()
        self.eyes_open()
        self.mouth_arc(smile=1)
        # blush circles
        r = 38
        self.canvas.create_oval(170, 330, 170 + 2 * r, 330 + 2 * r, fill=self.BLUSH, outline="")
        self.canvas.create_oval(620, 330, 620 + 2 * r, 330 + 2 * r, fill=self.BLUSH, outline="")
    
    def draw_surprised(self):
        self.clear()
        self.draw_face_base()
        self.eyes_open()
        # round mouth
        self.neon_oval(370, 380, 450, 460)
    
    def draw_angry(self):
        self.clear()
        self.draw_face_base()
        # angry brows
        # raise brows above eyes to avoid overlap
        self.neon_line(220, 120, 330, 155)
        self.neon_line(600, 120, 490, 155)
        self.eyes_open()
        self.mouth_arc(smile=-1)
    
    def draw_cry(self):
        self.clear()
        self.draw_face_base()
        # downward semi-circle eyebrows for cry expression
        self.neon_arc(220, 130, 360, 190, start=0, extent=180)
        self.neon_arc(460, 130, 600, 190, start=0, extent=180)
        self.eyes_open()
        self.mouth_arc(smile=-1)
        # three small tears for each eye, arranged vertically
        # left eye tears
        self.canvas.create_oval(280, 300, 300, 320, outline=self.TEAR, width=self.stroke)
        self.canvas.create_oval(280, 325, 300, 345, outline=self.TEAR, width=self.stroke)
        self.canvas.create_oval(280, 350, 300, 370, outline=self.TEAR, width=self.stroke)
        # right eye tears
        self.canvas.create_oval(540, 300, 560, 320, outline=self.TEAR, width=self.stroke)
        self.canvas.create_oval(540, 325, 560, 345, outline=self.TEAR, width=self.stroke)
        self.canvas.create_oval(540, 350, 560, 370, outline=self.TEAR, width=self.stroke)
    
    def draw_sleepy(self):
        self.clear()
        self.draw_face_base()
        self.eyes_closed()
        # wavy mouth - centered
        mid_y = 430
        x_start = 270  # centered under eyes
        amplitude = 14
        wave_len = 40
        points = []
        for i in range(8):
            x = x_start + i * wave_len
            y = mid_y + (amplitude if i % 2 else -amplitude)
            points.extend([x, y])
        # connect with short segments
        for i in range(0, len(points) - 2, 2):
            self.neon_line(points[i], points[i + 1], points[i + 2], points[i + 3])
        # Zs - positioned above and to the right of the right eye
        self.neon_line(650, 120, 680, 120)
        self.neon_line(680, 120, 650, 150)
        self.neon_line(650, 150, 680, 150)
        self.neon_line(670, 100, 690, 100)
        self.neon_line(690, 100, 670, 120)
        self.neon_line(670, 120, 690, 120)


def main():
    root = tk.Tk()
    app = EnhancedVoiceControlledFaceDisplay(root)
    root.mainloop()


if __name__ == "__main__":
    main()
