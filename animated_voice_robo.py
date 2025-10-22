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


class AnimatedExpression:
    """Handles smooth transitions between facial expressions"""
    
    def __init__(self, canvas):
        self.canvas = canvas
        self.current_expression = None
        self.target_expression = None
        self.animation_frame = 0
        self.animation_speed = 0.1  # Higher = faster animation
        self.animation_duration = 20  # Number of frames for transition
        
        # Animation state
        self.is_animating = False
        self.animation_thread = None
        
        # Expression parameters that will be animated
        self.eye_positions = {'left': (230, 170, 350, 290), 'right': (470, 170, 590, 290)}
        self.mouth_curve = 0  # 1 = happy, 0 = neutral, -1 = sad
        self.eyebrow_angle = 0  # 1 = raised, 0 = normal, -1 = lowered
        self.eye_openness = 1  # 1 = open, 0 = closed
        self.tear_visibility = 0  # 0 = none, 1 = visible
        self.blush_intensity = 0  # 0 = none, 1 = full blush
        self.heart_eyes = 0  # 0 = normal, 1 = heart shaped
        
    def start_transition(self, target_expression):
        """Start smooth transition to target expression"""
        if self.current_expression == target_expression:
            return
            
        self.target_expression = target_expression
        self.animation_frame = 0
        self.is_animating = True
        
        if self.animation_thread and self.animation_thread.is_alive():
            return
            
        self.animation_thread = threading.Thread(target=self._animate_loop, daemon=True)
        self.animation_thread.start()
    
    def _animate_loop(self):
        """Main animation loop"""
        while self.is_animating and self.animation_frame < self.animation_duration:
            progress = self.animation_frame / self.animation_duration
            
            # Smooth easing function (ease-in-out)
            eased_progress = self._ease_in_out(progress)
            
            # Interpolate parameters
            self._interpolate_parameters(eased_progress)
            
            # Redraw face
            self._draw_animated_face()
            
            self.animation_frame += 1
            time.sleep(self.animation_speed)
        
        # Set final state
        self.current_expression = self.target_expression
        self.is_animating = False
        self._apply_final_expression()
    
    def _ease_in_out(self, t):
        """Smooth easing function for natural animation"""
        return t * t * (3.0 - 2.0 * t)
    
    def _interpolate_parameters(self, progress):
        """Interpolate between current and target expression parameters"""
        if not self.target_expression:
            return
            
        # Define target parameters for each expression
        target_params = {
            'happy': {'mouth_curve': 1, 'eyebrow_angle': 0, 'eye_openness': 1, 'tear_visibility': 0, 'blush_intensity': 0, 'heart_eyes': 0},
            'sad': {'mouth_curve': -1, 'eyebrow_angle': 0, 'eye_openness': 1, 'tear_visibility': 0, 'blush_intensity': 0, 'heart_eyes': 0},
            'neutral': {'mouth_curve': 0, 'eyebrow_angle': 0, 'eye_openness': 1, 'tear_visibility': 0, 'blush_intensity': 0, 'heart_eyes': 0},
            'angry': {'mouth_curve': -1, 'eyebrow_angle': -1, 'eye_openness': 1, 'tear_visibility': 0, 'blush_intensity': 0, 'heart_eyes': 0},
            'cry': {'mouth_curve': -1, 'eyebrow_angle': -0.5, 'eye_openness': 1, 'tear_visibility': 1, 'blush_intensity': 0, 'heart_eyes': 0},
            'sleepy': {'mouth_curve': 0, 'eyebrow_angle': 0, 'eye_openness': 0, 'tear_visibility': 0, 'blush_intensity': 0, 'heart_eyes': 0},
            'surprised': {'mouth_curve': 0, 'eyebrow_angle': 0, 'eye_openness': 1, 'tear_visibility': 0, 'blush_intensity': 0, 'heart_eyes': 0},
            'laughing': {'mouth_curve': 1, 'eyebrow_angle': 0, 'eye_openness': 0, 'tear_visibility': 1, 'blush_intensity': 0, 'heart_eyes': 0},
            'lovely': {'mouth_curve': 1, 'eyebrow_angle': 0, 'eye_openness': 1, 'tear_visibility': 0, 'blush_intensity': 0, 'heart_eyes': 1},
            'blushing': {'mouth_curve': 1, 'eyebrow_angle': 0, 'eye_openness': 1, 'tear_visibility': 0, 'blush_intensity': 1, 'heart_eyes': 0}
        }
        
        current_params = {
            'mouth_curve': self.mouth_curve,
            'eyebrow_angle': self.eyebrow_angle,
            'eye_openness': self.eye_openness,
            'tear_visibility': self.tear_visibility,
            'blush_intensity': self.blush_intensity,
            'heart_eyes': self.heart_eyes
        }
        
        target = target_params.get(self.target_expression, current_params)
        
        # Interpolate each parameter
        for param in current_params:
            current_val = current_params[param]
            target_val = target[param]
            self.__dict__[param] = current_val + (target_val - current_val) * progress
    
    def _draw_animated_face(self):
        """Draw the face with current animated parameters"""
        # Colors
        BG = "#0f1220"
        NEON = "#5ce1ff"
        TEAR = "#6fd7ff"
        BLUSH = "#ff99cc"
        stroke = 16
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Draw eyes
        self._draw_animated_eyes(NEON, stroke)
        
        # Draw eyebrows
        self._draw_animated_eyebrows(NEON, stroke)
        
        # Draw mouth
        self._draw_animated_mouth(NEON, stroke)
        
        # Draw special effects
        if self.tear_visibility > 0:
            self._draw_animated_tears(TEAR, stroke)
        
        if self.blush_intensity > 0:
            self._draw_animated_blush(BLUSH)
        
        if self.heart_eyes > 0:
            self._draw_animated_heart_eyes(NEON, stroke)
    
    def _draw_animated_eyes(self, color, stroke):
        """Draw eyes with animation"""
        lx0, ly0, lx1, ly1 = self.eye_positions['left']
        rx0, ry0, rx1, ry1 = self.eye_positions['right']
        
        if self.eye_openness > 0.5:  # Eyes mostly open
            if self.heart_eyes > 0.5:  # Heart eyes
                self._draw_heart_eye(290, 230, 70 * self.heart_eyes, color, stroke)
                self._draw_heart_eye(530, 230, 70 * self.heart_eyes, color, stroke)
            else:  # Normal circular eyes
                self.canvas.create_oval(lx0, ly0, lx1, ly1, outline=color, width=stroke)
                self.canvas.create_oval(rx0, ry0, rx1, ry1, outline=color, width=stroke)
        else:  # Eyes closed
            y = (ly0 + ly1) / 2
            self.canvas.create_line(lx0 + 20, y, lx1 - 20, y, fill=color, width=stroke, capstyle=tk.ROUND)
            self.canvas.create_line(rx0 + 20, y, rx1 - 20, y, fill=color, width=stroke, capstyle=tk.ROUND)
    
    def _draw_animated_eyebrows(self, color, stroke):
        """Draw eyebrows with animation"""
        if self.eyebrow_angle < -0.3:  # Angry eyebrows
            # Angry brows - raised above eyes
            self.canvas.create_line(220, 120, 330, 155, fill=color, width=stroke, capstyle=tk.ROUND)
            self.canvas.create_line(600, 120, 490, 155, fill=color, width=stroke, capstyle=tk.ROUND)
        elif self.eyebrow_angle < -0.1:  # Sad eyebrows (slight downward curve)
            # Downward curved eyebrows for cry
            self.canvas.create_arc(220, 130, 360, 190, start=0, extent=180, style=tk.ARC, outline=color, width=stroke)
            self.canvas.create_arc(460, 130, 600, 190, start=0, extent=180, style=tk.ARC, outline=color, width=stroke)
    
    def _draw_animated_mouth(self, color, stroke):
        """Draw mouth with animation"""
        x0, y0, x1, y1 = 220, 340, 600, 520
        
        if abs(self.mouth_curve) < 0.1:  # Neutral mouth
            if self.target_expression == 'surprised':
                # Round mouth for surprise
                self.canvas.create_oval(370, 380, 450, 460, outline=color, width=stroke)
            else:
                # Straight line
                self.canvas.create_line(270, 430, 550, 430, fill=color, width=stroke, capstyle=tk.ROUND)
        elif self.mouth_curve > 0:  # Happy mouth
            if self.target_expression == 'laughing':
                # Big laughing mouth
                self.canvas.create_arc(250, 370, 570, 520, start=200, extent=140, style=tk.ARC, outline=color, width=stroke + 6)
                self.canvas.create_arc(250, 370, 570, 520, start=200, extent=140, style=tk.ARC, outline=color, width=stroke)
            else:
                # Normal smile
                self.canvas.create_arc(x0, y0, x1, y1, start=200, extent=140, style=tk.ARC, outline=color, width=stroke)
        else:  # Sad mouth
            self.canvas.create_arc(x0, y0 + 40, x1, y1 + 40, start=20, extent=140, style=tk.ARC, outline=color, width=stroke)
    
    def _draw_animated_tears(self, color, stroke):
        """Draw tears with animation"""
        alpha = self.tear_visibility
        
        if self.target_expression == 'laughing':
            # Tears of joy (small droplets)
            self.canvas.create_oval(200, 320, 230, 360, outline=color, width=int(stroke * alpha))
            self.canvas.create_oval(590, 320, 620, 360, outline=color, width=int(stroke * alpha))
        elif self.target_expression == 'cry':
            # Three small tears for each eye, arranged vertically
            # Left eye tears
            self.canvas.create_oval(280, 300, 300, 320, outline=color, width=int(stroke * alpha))
            self.canvas.create_oval(280, 325, 300, 345, outline=color, width=int(stroke * alpha))
            self.canvas.create_oval(280, 350, 300, 370, outline=color, width=int(stroke * alpha))
            # Right eye tears
            self.canvas.create_oval(540, 300, 560, 320, outline=color, width=int(stroke * alpha))
            self.canvas.create_oval(540, 325, 560, 345, outline=color, width=int(stroke * alpha))
            self.canvas.create_oval(540, 350, 560, 370, outline=color, width=int(stroke * alpha))
    
    def _draw_animated_blush(self, color):
        """Draw blush with animation"""
        alpha = self.blush_intensity
        r = int(38 * alpha)
        self.canvas.create_oval(170, 330, 170 + 2 * r, 330 + 2 * r, fill=color, outline="")
        self.canvas.create_oval(620, 330, 620 + 2 * r, 330 + 2 * r, fill=color, outline="")
    
    def _draw_animated_heart_eyes(self, color, stroke):
        """Draw heart eyes with animation"""
        alpha = self.heart_eyes
        self._draw_heart_eye(290, 230, int(70 * alpha), color, stroke)
        self._draw_heart_eye(530, 230, int(70 * alpha), color, stroke)
    
    def _draw_heart_eye(self, cx, cy, size, color, stroke):
        """Draw a single heart eye"""
        if size <= 0:
            return
        r = size // 2
        self.canvas.create_arc(cx - size, cy - size, cx, cy, start=45, extent=225, style=tk.ARC, outline=color, width=stroke)
        self.canvas.create_arc(cx, cy - size, cx + size, cy, start=-45, extent=225, style=tk.ARC, outline=color, width=stroke)
        self.canvas.create_line(cx - r, cy, cx, cy + r, fill=color, width=stroke, capstyle=tk.ROUND)
        self.canvas.create_line(cx + r, cy, cx, cy + r, fill=color, width=stroke, capstyle=tk.ROUND)
    
    def _apply_final_expression(self):
        """Apply the final expression without animation"""
        self._draw_animated_face()
        
        # Add special elements that don't animate
        if self.target_expression == 'sleepy':
            self._draw_sleep_elements()
    
    def _draw_sleep_elements(self):
        """Draw Z's for sleepy expression"""
        color = "#5ce1ff"
        stroke = 16
        # Zs - positioned above and to the right of the right eye
        self.canvas.create_line(650, 120, 680, 120, fill=color, width=stroke, capstyle=tk.ROUND)
        self.canvas.create_line(680, 120, 650, 150, fill=color, width=stroke, capstyle=tk.ROUND)
        self.canvas.create_line(650, 150, 680, 150, fill=color, width=stroke, capstyle=tk.ROUND)
        self.canvas.create_line(670, 100, 690, 100, fill=color, width=stroke, capstyle=tk.ROUND)
        self.canvas.create_line(690, 100, 670, 120, fill=color, width=stroke, capstyle=tk.ROUND)
        self.canvas.create_line(670, 120, 690, 120, fill=color, width=stroke, capstyle=tk.ROUND)
        
        # Wavy mouth for sleepy
        mid_y = 430
        x_start = 270
        amplitude = 14
        wave_len = 40
        points = []
        for i in range(8):
            x = x_start + i * wave_len
            y = mid_y + (amplitude if i % 2 else -amplitude)
            points.extend([x, y])
        for i in range(0, len(points) - 2, 2):
            self.canvas.create_line(points[i], points[i + 1], points[i + 2], points[i + 3], 
                                  fill=color, width=stroke, capstyle=tk.ROUND)


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
            print(f"Error extracting features: {e}")
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
                if isinstance(results, list) and len(results) > 0:
                    for result in results[0]:  # Fix: access first element properly
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


class AnimatedVoiceControlledFaceDisplay:
    """Animated voice-controlled robot face with smooth transitions"""
    
    # Colors and sizes
    BG = "#0f1220"
    NEON = "#5ce1ff"
    TEAR = "#6fd7ff"
    BLUSH = "#ff99cc"
    
    def __init__(self, root):
        self.root = root
        self.root.title("Animated Voice-Controlled Robo Face")
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
        
        # Animation system
        self.animation = AnimatedExpression(self.canvas)
        
        # Calibrate microphone
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        
        self.setup_gui()
        self.setup_voice_control()
        
        # Initial expression
        self.animation.start_transition('happy')
    
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
            text="Start Animated Voice Control",
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
            text="Animated Voice Control: OFF",
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
            ("1\nHappy", lambda: self.animation.start_transition('happy')),
            ("2\nSad", lambda: self.animation.start_transition('sad')),
            ("3\nNeutral", lambda: self.animation.start_transition('neutral')),
            ("4\nLovely", lambda: self.animation.start_transition('lovely')),
            ("5\nLaugh", lambda: self.animation.start_transition('laughing')),
            ("6\nBlush", lambda: self.animation.start_transition('blushing')),
            ("7\nSurprise", lambda: self.animation.start_transition('surprised')),
            ("8\nAngry", lambda: self.animation.start_transition('angry')),
            ("9\nCry", lambda: self.animation.start_transition('cry')),
            ("0\nSleep", lambda: self.animation.start_transition('sleepy')),
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
        self.root.bind("1", lambda _e: self.animation.start_transition('happy'))
        self.root.bind("2", lambda _e: self.animation.start_transition('sad'))
        self.root.bind("3", lambda _e: self.animation.start_transition('neutral'))
        self.root.bind("4", lambda _e: self.animation.start_transition('lovely'))
        self.root.bind("5", lambda _e: self.animation.start_transition('laughing'))
        self.root.bind("6", lambda _e: self.animation.start_transition('blushing'))
        self.root.bind("7", lambda _e: self.animation.start_transition('surprised'))
        self.root.bind("8", lambda _e: self.animation.start_transition('angry'))
        self.root.bind("9", lambda _e: self.animation.start_transition('cry'))
        self.root.bind("0", lambda _e: self.animation.start_transition('sleepy'))
    
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
                            
                            # Update expression based on emotion with animation
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
            self.voice_button.config(text="Stop Animated Voice Control")
            self.status_label.config(text="Animated Voice Control: ON")
            self.confidence_label.config(text="Confidence: --")
        else:
            self.voice_button.config(text="Start Animated Voice Control")
            self.status_label.config(text="Animated Voice Control: OFF")
            self.confidence_label.config(text="Confidence: --")
    
    def update_expression_from_emotion(self, emotion_result):
        """Update expression based on detected emotion with smooth animation"""
        emotion = emotion_result['emotion']
        confidence = emotion_result['confidence']
        audio_emotion = emotion_result.get('audio_emotion', 'unknown')
        text_emotion = emotion_result.get('text_emotion', 'unknown')
        
        # Start animated transition
        self.animation.start_transition(emotion)
        
        # Update status with detailed information
        status_text = f"Animated Voice Control: ON - Detected: {emotion.upper()}"
        if audio_emotion != 'unknown' or text_emotion != 'unknown':
            status_text += f" (Audio: {audio_emotion}, Text: {text_emotion})"
        
        self.status_label.config(text=status_text)
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}")


def main():
    root = tk.Tk()
    app = AnimatedVoiceControlledFaceDisplay(root)
    root.mainloop()


if __name__ == "__main__":
    main()

