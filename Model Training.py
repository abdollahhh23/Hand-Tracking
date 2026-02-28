import os
import pandas as pd
import cv2
import mediapipe as mp
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle


Dataset= r"C:\Users\User\PyProject\hand_landmarks_data.csv"
df = pd.read_csv(Dataset)

X = df.iloc[:, :-1].values
y = df['label'].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

model = svm.SVC(kernel='linear', C=1.0) 
model.fit(x_train, y_train)

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
    
print("Model trained and saved as model.p!")