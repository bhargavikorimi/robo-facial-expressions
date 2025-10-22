import tkinter as tk
from math import cos, sin, pi
import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import threading # Use threading to prevent GUI from freezing while listening

# --- Sentiment Analysis and Speech Recognition Functions ---

# Initialize the VADER analyzer once
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text: str) -> str:
    """Analyzes text sentiment using VADER and returns 'positive', 'negative', or 'neutral'."""
    if not text:
        return "neutral"
    
    vs = analyzer.polarity_scores(text)
    
    # VADER Compound Score Thresholds (Standard)
    # Compound score > 0.05 is positive
    # Compound score < -0.05 is negative
    if vs['compound'] >= 0.05:
        # Appreciation/Encouraging -> Happy
        return "positive"  
    elif vs['compound'] <= -0.05:
        # Complaining/Angry -> Sad
        return "negative" 
    else:
        # Normal tone -> Neutral
        return "neutral" 

def listen_and_analyze(callback_func):
    """Listens to the microphone, transcribes speech, and calls a callback function with the result."""
    r = sr.Recognizer()
    
    # Use a thread to perform the listening/processing so the Tkinter GUI doesn't freeze
    def background_listen():
        with sr.Microphone() as source:
            print("Listening...")
            r.adjust_for_ambient_noise(source, duration=0.5)
            try:
                # Listen for up to 5 seconds of speech
                audio = r.listen(source, timeout=5) 
                print("Processing...")
                
                # Use Google Web Speech API for transcription
                text = r.recognize_google(audio)
                
                # Analyze sentiment
                sentiment = get_sentiment(text)
                print(f"Recognized: '{text}' | Sentiment: {sentiment}")

                # Call the Tkinter function to update the face
                callback_func(sentiment) 
                
            except sr.WaitTimeoutError:
                print("No speech detected.")
                # Automatically revert to neutral or keep the current face
                # For this example, we keep the current face.
            except sr.UnknownValueError:
                print("Could not understand audio.")
                callback_func("neutral") # Fallback to neutral on failure to understand
            except sr.RequestError as e:
                print(f"Could not request results from Google SR service; {e}")
                callback_func("neutral")

    # Start the listening process in a new thread
    threading.Thread(target=background_listen).start()


# --- FaceDisplay Class (Modified) ---

class FaceDisplay:
    """Simple canvas-based face renderer with voice-controlled expressions."""

    # Colors and sizes tuned to resemble neon-blue on a dark screen
    BG = "#0f1220"
    NEON = "#5ce1ff"
    TEAR = "#6fd7ff"
    BLUSH = "#ff99cc"
    
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Robo Face Expressions (Voice Control Enabled)")
        self.root.configure(bg=self.BG)

        self.canvas_width = 820
        self.canvas_height = 540
        self.stroke = 16

        self.canvas = tk.Canvas(
            root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg=self.BG,
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, columnspan=10, padx=12, pady=(12, 4))
        
        # Original Buttons 1-9,0 (Keep for manual control)
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
        for idx, (label, handler) in enumerate(btn_specs):
            b = tk.Button(
                root,
                text=label,
                width=8,
                bg="#162036",
                fg=self.NEON,
                activebackground="#1e2a46",
                activeforeground=self.NEON,
                relief=tk.FLAT,
                command=handler,
            )
            b.grid(row=1, column=idx, padx=4, pady=(0, 4))
            
        # NEW Voice Control Button
        self.voice_btn = tk.Button(
            root,
            text="START VOICE COMMAND",
            width=40,
            bg="#2a52be", # A distinct color for the new feature
            fg="white",
            font=('Helvetica', 12, 'bold'),
            relief=tk.RAISED,
            command=self.start_voice_command_thread,
        )
        self.voice_btn.grid(row=2, column=0, columnspan=10, pady=(4, 12)) 

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

        # Initial expression
        self.draw_neutral() # Start Neutral for voice testing

    # --- NEW VOICE CONTROL METHODS ---
    
    def start_voice_command_thread(self):
        """Disables the button and starts the voice listening process."""
        self.voice_btn.config(text="Listening...", state=tk.DISABLED, bg="#990000")
        # Pass self.update_face_from_sentiment as the callback
        listen_and_analyze(self.update_face_from_sentiment) 

    def update_face_from_sentiment(self, sentiment: str):
        """Called by the background thread to update the GUI based on sentiment."""
        # Use root.after to safely update the GUI from a non-Tkinter thread
        self.root.after(0, lambda: self._handle_face_update(sentiment))

    def _handle_face_update(self, sentiment: str):
        """Core logic to switch the expression and re-enable the button."""
        if sentiment == "positive":
            self.draw_happy()
        elif sentiment == "negative":
            self.draw_sad()
        else: # neutral or no speech detected/error
            self.draw_neutral()
            
        # Re-enable the button regardless of the result
        self.voice_btn.config(text="START VOICE COMMAND", state=tk.NORMAL, bg="#2a52be")

    # --- Original Drawing Helpers & Expressions (Unchanged) ---
    def clear(self) -> None:
        self.canvas.delete("all")

    def neon_oval(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self.canvas.create_oval(
            x0,
            y0,
            x1,
            y1,
            outline=self.NEON,
            width=self.stroke,
        )

    def neon_arc(self, x0: int, y0: int, x1: int, y1: int, start: float, extent: float) -> None:
        self.canvas.create_arc(
            x0,
            y0,
            x1,
            y1,
            start=start,
            extent=extent,
            style=tk.ARC,
            outline=self.NEON,
            width=self.stroke,
        )

    def neon_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self.canvas.create_line(
            x0,
            y0,
            x1,
            y1,
            fill=self.NEON,
            width=self.stroke,
            capstyle=tk.ROUND,
        )

    def heart(self, cx: int, cy: int, size: int) -> None:
        # Draw a heart using two arcs and a V line
        r = size // 2
        self.neon_arc(cx - size, cy - size, cx, cy, start=45, extent=225)
        self.neon_arc(cx, cy - size, cx + size, cy, start=-45, extent=225)
        self.neon_line(cx - r, cy, cx, cy + r)
        self.neon_line(cx + r, cy, cx, cy + r)

    def draw_face_base(self) -> None:
        # Eye positions
        self.left_eye = (230, 170, 350, 290)
        self.right_eye = (470, 170, 590, 290)

    def eyes_open(self) -> None:
        lx0, ly0, lx1, ly1 = self.left_eye
        rx0, ry0, rx1, ry1 = self.right_eye
        self.neon_oval(lx0, ly0, lx1, ly1)
        self.neon_oval(rx0, ry0, rx1, ry1)

    def eyes_closed(self) -> None:
        # draw two flat lines as closed eyes
        lx0, ly0, lx1, _ = self.left_eye
        rx0, ry0, rx1, _ = self.right_eye
        y = (ly0 + 40)
        self.neon_line(lx0 + 20, y, lx1 - 20, y)
        self.neon_line(rx0 + 20, y, rx1 - 20, y)

    def mouth_arc(self, smile: int) -> None:
        # smile >0 for up-curve, <0 for down-curve, 0 for neutral
        x0, y0, x1, y1 = 220, 340, 600, 520
        if smile > 0:
            self.neon_arc(x0, y0, x1, y1, start=200, extent=140)
        elif smile < 0:
            self.neon_arc(x0, y0 + 40, x1, y1 + 40, start=20, extent=140)
        else:
            self.neon_line(270, 430, 550, 430)

    def draw_happy(self) -> None:
        self.clear()
        self.draw_face_base()
        self.eyes_open()
        self.mouth_arc(smile=1)

    def draw_sad(self) -> None:
        self.clear()
        self.draw_face_base()
        self.eyes_open()
        self.mouth_arc(smile=-1)

    def draw_neutral(self) -> None:
        self.clear()
        self.draw_face_base()
        self.eyes_open()
        self.mouth_arc(smile=0)

    def draw_lovely(self) -> None:
        self.clear()
        self.draw_face_base()
        self.heart(290, 230, 70)
        self.heart(530, 230, 70)
        self.mouth_arc(smile=1)

    def draw_laughing(self) -> None:
        self.clear()
        self.draw_face_base()
        self.eyes_closed()
        x0, y0, x1, y1 = 250, 370, 570, 520
        self.canvas.create_arc(
            x0,
            y0,
            x1,
            y1,
            start=200,
            extent=140,
            style=tk.ARC,
            outline=self.NEON,
            width=self.stroke + 6,
        )
        self.neon_arc(x0, y0, x1, y1, start=200, extent=140)
        self.canvas.create_oval(200, 320, 230, 360, outline=self.TEAR, width=self.stroke)
        self.canvas.create_oval(590, 320, 620, 360, outline=self.TEAR, width=self.stroke)

    def draw_blushing(self) -> None:
        self.clear()
        self.draw_face_base()
        self.eyes_open()
        self.mouth_arc(smile=1)
        r = 38
        self.canvas.create_oval(170, 330, 170 + 2 * r, 330 + 2 * r, fill=self.BLUSH, outline="")
        self.canvas.create_oval(620, 330, 620 + 2 * r, 330 + 2 * r, fill=self.BLUSH, outline="")

    def draw_surprised(self) -> None:
        self.clear()
        self.draw_face_base()
        self.eyes_open()
        self.neon_oval(370, 380, 450, 460)

    def draw_angry(self) -> None:
        self.clear()
        self.draw_face_base()
        self.neon_line(220, 120, 330, 155)
        self.neon_line(600, 120, 490, 155)
        self.eyes_open()
        self.mouth_arc(smile=-1)

    def draw_cry(self) -> None:
        self.clear()
        self.draw_face_base()
        self.neon_arc(220, 130, 360, 190, start=0, extent=180)
        self.neon_arc(460, 130, 600, 190, start=0, extent=180)
        self.eyes_open()
        self.mouth_arc(smile=-1)
        self.canvas.create_oval(280, 300, 300, 320, outline=self.TEAR, width=self.stroke)
        self.canvas.create_oval(280, 325, 300, 345, outline=self.TEAR, width=self.stroke)
        self.canvas.create_oval(280, 350, 300, 370, outline=self.TEAR, width=self.stroke)
        self.canvas.create_oval(540, 300, 560, 320, outline=self.TEAR, width=self.stroke)
        self.canvas.create_oval(540, 325, 560, 345, outline=self.TEAR, width=self.stroke)
        self.canvas.create_oval(540, 350, 560, 370, outline=self.TEAR, width=self.stroke)

    def draw_sleepy(self) -> None:
        self.clear()
        self.draw_face_base()
        self.eyes_closed()
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
            self.neon_line(points[i], points[i + 1], points[i + 2], points[i + 3])
        self.neon_line(650, 120, 680, 120)
        self.neon_line(680, 120, 650, 150)
        self.neon_line(650, 150, 680, 150)
        self.neon_line(670, 100, 690, 100)
        self.neon_line(690, 100, 670, 120)
        self.neon_line(670, 120, 690, 120)


def main() -> None:
    root = tk.Tk()
    app = FaceDisplay(root)
    root.mainloop()


if __name__ == "__main__":
    main()