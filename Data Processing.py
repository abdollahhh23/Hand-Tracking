import os
import pandas as pd
import cv2
import mediapipe as mp

#mediapipe hands initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

Dataset=' Folder path '
data=[]
label=[]

for directories in os.listdir(Dataset):
    if not os.path.isdir(os.path.join(Dataset, directories)):
        continue

    print("Processing Letter:, {directories}")

    for image_path in os.listdir(os.path.join(Dataset, directories)):
        
        data_aux = []
        image=cv2.imread(os.path.join(Dataset, directories, image_path))
        image_rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results= hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x)
            data_aux.append(y)

        data.append(data_aux)
        label.append(directories)


df= pd.DataFrame(data)
df['label']= label
df.to_csv('hand_landmarks_data.csv', index= False)
print("Data Extraction Complete")