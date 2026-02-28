import os
import cv2
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

DATA_DIR = r"C:\Users\User\Desktop\ASL Data\Train" 
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(folder_path):
        continue
        
    print(f"Processing Letter: {dir_}")
    
    for img_path in os.listdir(folder_path):
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = cv2.imread(os.path.join(folder_path, img_path))
        if img is None: continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # We focus on the first hand detected
            hand_landmarks = results.multi_hand_landmarks[0]
            data_aux = []
            
            # WRIST-CENTRIC NORMALIZATION
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y

            for i in range(21):
                # Store coordinates as distance from wrist
                data_aux.append(hand_landmarks.landmark[i].x - wrist_x)
                data_aux.append(hand_landmarks.landmark[i].y - wrist_y)
            
            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(dir_)

# Save to CSV
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv('hand_landmarks_data.csv', index=False)
print(" Data extraction complete! Landmarks saved to hand_landmarks_data.csv")