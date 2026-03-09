import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
from textblob import TextBlob
from tensorflow.keras.models import load_model

# Load trained gesture model
model = load_model("gesture_model.h5")

# Word labels (must match training order)
labels = np.load("class_labels.npy")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="Dynamic Sign Language Detection", layout="wide")

st.title("🖐️ Real-Time Sign Language Recognition")
st.markdown("Perform a gesture for **1–2 seconds** to detect a word.")

st.sidebar.header("⚙️ Controls")
run = st.sidebar.checkbox("Run Webcam", value=True)

col1, col2 = st.columns([3,1])

with col1:
    FRAME_WINDOW = st.empty()

with col2:
    st.subheader("📊 Prediction")
    pred_text = st.empty()
    confidence_bar = st.progress(0)

    st.markdown("---")
    st.subheader("📝 Sentence")
    sentence_display = st.empty()

# session memory
if "sentence" not in st.session_state:
    st.session_state.sentence = ""

hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

# sequence buffer for dynamic gesture
sequence = []
sequence_length = 30

last_prediction_time = 0


def autocorrect_sentence():
    if st.session_state.sentence.strip():
        corrected = str(TextBlob(st.session_state.sentence.lower()).correct())
        st.session_state.sentence = corrected.capitalize()


st.sidebar.button("🔁 Autocorrect", on_click=autocorrect_sentence)

while run:

    ret, frame = cap.read()

    if not ret:
        st.warning("⚠️ Cannot access webcam")
        break

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    label = "No gesture"
    prob = 0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            row = []

            for lm in hand_landmarks.landmark:
                row += [lm.x, lm.y, lm.z]

            sequence.append(row)

            if len(sequence) > sequence_length:
                sequence.pop(0)

            mp_drawing.draw_landmarks(frame,
                                      hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS)

            # Predict when sequence full
            if len(sequence) == sequence_length:

                input_data = np.array(sequence)
                input_data = np.expand_dims(input_data, axis=0)

                prediction = model.predict(input_data)[0]

                idx = np.argmax(prediction)

                label = labels[idx]
                prob = prediction[idx]

                current_time = time.time()

                if prob > 0.8 and current_time - last_prediction_time > 2:

                    if label == "space":
                        st.session_state.sentence += " "
                    else:
                        st.session_state.sentence += label + " "

                    last_prediction_time = current_time

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                       caption="Live Feed",
                       use_container_width=True)

    pred_text.markdown(f"### 🤟 Predicted Word: **{label}**")

    confidence_bar.progress(int(prob*100))

    sentence_display.markdown(f"### ✍️ {st.session_state.sentence}")

cap.release()