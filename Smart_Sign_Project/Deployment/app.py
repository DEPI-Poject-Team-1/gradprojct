import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import deque, Counter
import gradio as gr
import time
import os

# Try different import methods for TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.resnet50 import preprocess_input
    print("✅ TensorFlow imported successfully")
except ImportError as e:
    print(f"❌ TensorFlow import error: {e}")
    # Fallback - try basic TensorFlow
    import tensorflow as tf
    from tensorflow import keras
    preprocess_input = tf.keras.applications.resnet50.preprocess_input

# ---------------- CONFIG ----------------
MODEL_PATH = "fixed_model.keras"
CONF_THRESHOLD = 0.6
SMOOTH_WINDOW = 8
MARGIN = 20

CLASS_LABELS = {
     0:'A',  1:'B',  2:'C',  3:'D',  4:'E',  5:'F',  6:'G',  7:'H',  8:'I',  9:'J',
    10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T',
    20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z', 26:'del', 27:'nothing', 28:'space'
}

# ---------------- LOAD MODEL WITH ERROR HANDLING ----------------
print("🔹 Loading model...")
model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded with tf.keras")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    try:
        # Try alternative loading
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded with direct load_model")
    except Exception as e2:
        print(f"❌ All loading methods failed: {e2}")
        # Create a dummy model for testing
        class DummyModel:
            def predict(self, x):
                return np.random.rand(1, len(CLASS_LABELS))
            @property
            def input_shape(self):
                return (None, 224, 224, 3)
        model = DummyModel()
        print("⚠️ Using dummy model for testing")

IMG_H, IMG_W, IMG_C = model.input_shape[1:4]
print(f"✅ Model input shape: ({IMG_H}x{IMG_W}x{IMG_C})")

# ---------------- INIT MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)

# ---------------- SMOOTHING BUFFER ----------------
pred_history = deque(maxlen=SMOOTH_WINDOW)

# ---------------- GLOBAL VARIABLES ----------------
ptime = 0
is_streaming = False
sentence_buffer = []

# ---------------- GRADIO PREDICTION FUNCTION ----------------
def process_frame(frame):
    global ptime, is_streaming
    
    if frame is None or not is_streaming:
        return None, 0, 0, "--", ""

    try:
        # Convert frame
        if len(frame.shape) == 3:
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        frame = cv.flip(frame, 1)
        h, w = frame.shape[:2]

        # Gray frame output
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_frame = cv.cvtColor(gray_frame, cv.COLOR_GRAY2BGR)

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(rgb)

        pred_label, conf = "nothing", 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Bounding box
                x = [lm.x for lm in hand_landmarks.landmark]
                y = [lm.y for lm in hand_landmarks.landmark]
                x_min, y_min = int(min(x) * w), int(min(y) * h)
                x_max, y_max = int(max(x) * w), int(max(y) * h)

                # Apply margin
                x_min = max(0, x_min - MARGIN)
                y_min = max(0, y_min - MARGIN)
                x_max = min(w, x_max + MARGIN)
                y_max = min(h, y_max + MARGIN)

                hand_roi = frame[y_min:y_max, x_min:x_max]
                if hand_roi.size == 0:
                    continue

                # Preprocess for ResNet50
                hand_img = cv.resize(hand_roi, (IMG_W, IMG_H))
                hand_img = cv.cvtColor(hand_img, cv.COLOR_BGR2RGB)
                img_arr = preprocess_input(np.expand_dims(hand_img.astype("float32"), axis=0))

                # Prediction
                preds = model.predict(img_arr, verbose=0)[0]
                conf = float(np.max(preds))
                label_idx = int(np.argmax(preds))
                pred_label = CLASS_LABELS.get(label_idx, "unknown") if conf >= CONF_THRESHOLD else "nothing"

                pred_history.append(pred_label)

                # Drawing
                cv.rectangle(gray_frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
                cv.putText(gray_frame, f"{pred_label} ({conf*100:.1f}%)",
                           (x_min + 5, y_min - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                # Draw hand connections
                for con in mp_hands.HAND_CONNECTIONS:
                    s, e = con
                    s_lm, e_lm = hand_landmarks.landmark[s], hand_landmarks.landmark[e]
                    cv.line(gray_frame,
                            (int(s_lm.x*w), int(s_lm.y*h)),
                            (int(e_lm.x*w), int(e_lm.y*h)),
                            (255,255,255), 2)

                # Draw landmarks
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    cv.circle(gray_frame, (cx, cy), 4, (0,0,255), -1)

        # Smoothed prediction
        stable_label = Counter(pred_history).most_common(1)[0][0] if pred_history else "nothing"
        cv.putText(gray_frame, f"Stable: {stable_label}",
                   (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # FPS
        ctime = time.time()
        fps = 1 / (ctime - ptime) if ptime else 0
        ptime = ctime
        cv.putText(gray_frame, f"FPS: {fps:.1f}", (10, h-20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        
        # Add streaming status
        status_color = (0, 255, 0) if is_streaming else (0, 0, 255)
        status_text = "LIVE" if is_streaming else "STOPPED"
        cv.putText(gray_frame, f"Status: {status_text}", (10, h-50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Update sentence
        current_sentence = update_sentence(stable_label)

        return cv.cvtColor(gray_frame, cv.COLOR_BGR2RGB), round(fps, 1), round(conf*100, 1), stable_label, current_sentence
    
    except Exception as e:
        print(f"Error in process_frame: {e}")
        # Return a black frame with error message
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv.putText(error_frame, f"Error: {str(e)}", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return cv.cvtColor(error_frame, cv.COLOR_BGR2RGB), 0, 0, "error", ""

# ---------------- SENTENCE FUNCTIONS ----------------
def update_sentence(prediction):
    global sentence_buffer
    
    if prediction in ['del', 'space', 'nothing']:
        if prediction == 'del' and sentence_buffer:
            sentence_buffer.pop()
        elif prediction == 'space':
            sentence_buffer.append(' ')
        return ''.join(sentence_buffer) if sentence_buffer else ""
    
    # Only add letters if different from last prediction
    if not sentence_buffer or sentence_buffer[-1] != prediction:
        sentence_buffer.append(prediction)
    
    return ''.join(sentence_buffer)

def clear_sentence():
    global sentence_buffer
    sentence_buffer = []
    return ""

# ---------------- BUTTON FUNCTIONS ----------------
def start_webcam():
    global is_streaming
    is_streaming = True
    print("🎥 Webcam started")
    return (
        gr.update(interactive=False),  # start_btn
        gr.update(interactive=True),   # stop_btn
        "🟢 Streaming... Show your hand to the camera!",
        "🟢 LIVE"
    )

def stop_webcam():
    global is_streaming
    is_streaming = False
    print("🛑 Webcam stopped")
    # Clear prediction history when stopping
    pred_history.clear()
    return (
        gr.update(interactive=True),   # start_btn
        gr.update(interactive=False),  # stop_btn
        "🔴 Streaming stopped. Click 'Start Detection' to begin.",
        "🔴 STOPPED"
    )

# ---------------- GRADIO UI (Gradio 6.0.1 Compatible) ----------------
with gr.Blocks() as demo:
    
    # Header Section
    gr.Markdown("""
    # 🤟 Smart Sign - ASL Detection
    **Real-time American Sign Language Recognition with AI**
    """)
    
    # Main Content Area
    with gr.Row():
        # Left Panel - Controls and Stats
        with gr.Column(scale=1):
            
            # Status Card
            gr.Markdown("### 📊 Session Status")
            status_display = gr.Textbox(
                value="🔴 Click 'Start Detection' to begin",
                label="Status",
                interactive=False
            )
            status_indicator = gr.Textbox(
                value="🔴 STOPPED",
                label="Live Status",
                interactive=False
            )
            
            # Control Panel
            gr.Markdown("### 🎮 Controls")
            with gr.Row():
                start_btn = gr.Button("▶️ Start Detection", variant="primary")
                stop_btn = gr.Button("⏹️ Stop", variant="stop", interactive=False)
            
            with gr.Row():
                reset_btn = gr.Button("🔄 Reset")
                clear_btn = gr.Button("🗑️ Clear Text")
            
            # Real-time Statistics
            gr.Markdown("### 📈 Live Statistics")
            with gr.Row():
                fps_display = gr.Number(label="FPS", value=0, interactive=False)
                confidence_display = gr.Number(label="Confidence %", value=0, interactive=False)
            
            detected_letter = gr.Textbox(label="Detected Letter", value="--", interactive=False)
            
            # Sentence Builder
            gr.Markdown("### 📝 Sentence Builder")
            sentence_output = gr.Textbox(
                label="Translated Text",
                lines=3,
                placeholder="Your translated text will appear here...",
                interactive=False
            )
    
        # Right Panel - Camera Feeds
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("🎥 Live Detection"):
                    with gr.Row():
                        webcam_input = gr.Image(
                            label="Webcam Feed",
                            streaming=True,
                            interactive=True
                        )
                        output_image = gr.Image(
                            label="Processed Output",
                            interactive=False
                        )
                
                with gr.TabItem("📚 ASL Guide"):
                    gr.Markdown("""
                    ### ASL Alphabet Reference
                    
                    **Letters A-Z Supported**
                    - Show clear hand signs to camera
                    - Ensure good lighting
                    - Keep hand within frame
                    - Make distinct gestures
                    
                    **Special Commands:**
                    - **space**: Add space between words
                    - **del**: Delete last character
                    - **nothing**: No hand detected
                    """)
    
    # Instructions Section
    gr.Markdown("""
    ## 📖 How to Use
    
    1. **Click 'Start Detection'** to begin
    2. **Allow camera access** when prompted  
    3. **Show your hand** clearly in frame
    4. **View real-time results** and build sentences
    5. **Click 'Stop'** when finished
    
    *Built with TensorFlow, MediaPipe, and Gradio*
    """)
    
    # ---------------- EVENT HANDLERS ----------------
    
    # Button actions
    start_btn.click(
        fn=start_webcam,
        inputs=[],
        outputs=[start_btn, stop_btn, status_display, status_indicator]
    )
    
    stop_btn.click(
        fn=stop_webcam,
        inputs=[],
        outputs=[start_btn, stop_btn, status_display, status_indicator]
    )
    
    clear_btn.click(
        fn=clear_sentence,
        inputs=[],
        outputs=[sentence_output]
    )
    
    reset_btn.click(
        fn=stop_webcam,  # Stop and reset
        inputs=[],
        outputs=[start_btn, stop_btn, status_display, status_indicator]
    )
    
    # Stream processing with multiple outputs
    webcam_input.stream(
        fn=process_frame,
        inputs=[webcam_input],
        outputs=[output_image, fps_display, confidence_display, detected_letter, sentence_output]
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True
    )