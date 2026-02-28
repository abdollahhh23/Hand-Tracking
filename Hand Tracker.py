import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import Counter

# 1. Load the model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: 'model.p' not found. Run your training script first!")
    exit()

# 2. Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Buffer for temporal smoothing (to prevent flickering)
prediction_history = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        data_aux = []
        # WRIST-CENTRIC NORMALIZATION (Must match extraction script)
        wrist_x = hand_landmarks.landmark[0].x
        wrist_y = hand_landmarks.landmark[0].y

        for i in range(21):
            data_aux.append(hand_landmarks.landmark[i].x - wrist_x)
            data_aux.append(hand_landmarks.landmark[i].y - wrist_y)

        # Predict
        prediction = model.predict([np.asarray(data_aux)])
        char = str(prediction[0])

        # Smooth the result
        prediction_history.append(char)
        if len(prediction_history) > 10:
            prediction_history.pop(0)
        
        stable_char = Counter(prediction_history).most_common(1)[0][0]

        # Display Result
        cv2.putText(frame, f"Letter: {stable_char}", (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('Sign Language Decoder', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()