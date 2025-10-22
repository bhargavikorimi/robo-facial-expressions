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


class VoiceEmotionDetector:
    """Detects emotion from voice using audio features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        # Simple emotion mapping based on audio features
        self.emotion_thresholds = {
            'angry': {'pitch_mean': 180, 'energy_mean': 0.3, 'tempo': 2.0},
            'happy': {'pitch_mean': 200, 'energy_mean': 0.4, 'tempo': 1.8},
            'sad': {'pitch_mean': 120, 'energy_mean': 0.15, 'tempo': 1.2},
            'neutral': {'pitch_mean': 150, 'energy_mean': 0.25, 'tempo': 1.5}
        }
    
    def extract_features(self, audio_data, sample_rate):
        """Extract audio features for emotion detection"""
        try:
            # Convert to float32 for librosa - handle different data types
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
            
            # Extract pitch (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 150
            
            # Extract energy
            energy = librosa.feature.rms(y=audio_data)[0]
            energy_mean = np.mean(energy)
            
            # Extract tempo
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            
            return {
                'pitch_mean': pitch_mean,
                'energy_mean': energy_mean,
                'tempo': tempo
            }
        except Exception as e:
            print(f"Error extracting features: {e}")
            return {'pitch_mean': 150, 'energy_mean': 0.25, 'tempo': 1.5}
    
    def detect_emotion(self, audio_data, sample_rate):
        """Detect emotion from audio features"""
        features = self.extract_features(audio_data, sample_rate)
        
        # Simple rule-based emotion detection
        pitch = features['pitch_mean']
        energy = features['energy_mean']
        tempo = features['tempo']
        
        # Angry: High pitch, high energy, fast tempo
        if pitch > 180 and energy > 0.3 and tempo > 2.0:
            return 'angry'
        # Happy: High pitch, high energy, moderate tempo
        elif pitch > 200 and energy > 0.4:
            return 'happy'
        # Sad: Low pitch, low energy, slow tempo
        elif pitch < 130 and energy < 0.2 and tempo < 1.3:
            return 'sad'
        # Neutral: Medium values
        else:
            return 'neutral'


class VoiceControlledFaceDisplay:
    """Voice-controlled robot face with emotion detection"""
    
    # Colors and sizes
    BG = "#0f1220"
    NEON = "#5ce1ff"
    TEAR = "#6fd7ff"
    BLUSH = "#ff99cc"
    
    def __init__(self, root):
        self.root = root
        self.root.title("Voice-Controlled Robo Face")
        self.root.configure(bg=self.BG)
        
        self.canvas_width = 820
        self.canvas_height = 540
        self.stroke = 16
        
        # Voice control setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.emotion_detector = VoiceEmotionDetector()
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
        """Setup the GUI components"""
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
            text="Start Voice Control",
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
        
        # Status label
        self.status_label = tk.Label(
            control_frame,
            text="Voice Control: OFF",
            bg=self.BG,
            fg=self.NEON,
            font=("Arial", 12)
        )
        self.status_label.pack(side=tk.LEFT, padx=20)
        
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
        """Setup voice recognition"""
        def listen_continuously():
            while True:
                if self.is_listening:
                    try:
                        with self.microphone as source:
                            # Listen for audio
                            audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                            
                            # Get raw audio data for emotion detection
                            audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
                            sample_rate = audio.sample_rate
                            
                            # Detect emotion
                            emotion = self.emotion_detector.detect_emotion(audio_data, sample_rate)
                            
                            # Update expression based on emotion
                            self.root.after(0, lambda: self.update_expression_from_emotion(emotion))
                            
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
            self.voice_button.config(text="Stop Voice Control")
            self.status_label.config(text="Voice Control: ON - Speak to change expressions")
        else:
            self.voice_button.config(text="Start Voice Control")
            self.status_label.config(text="Voice Control: OFF")
    
    def update_expression_from_emotion(self, emotion):
        """Update expression based on detected emotion"""
        emotion_map = {
            'angry': self.draw_angry,
            'happy': self.draw_happy,
            'sad': self.draw_sad,
            'neutral': self.draw_neutral
        }
        
        if emotion in emotion_map:
            emotion_map[emotion]()
            self.status_label.config(text=f"Voice Control: ON - Detected: {emotion.upper()}")
    
    # --------------------- Drawing helpers ---------------------
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
    
    # --------------------- Expressions ---------------------
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
    app = VoiceControlledFaceDisplay(root)
    root.mainloop()


if __name__ == "__main__":
    main()
