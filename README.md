Hand Tracking
A computer vision system that detects and tracks hand landmarks in real-time. This project serves as the foundational data collection and processing engine for gesture recognition applications.

Overview
This system uses a webcam to identify 21 specific hand landmarks. It includes tools for capturing coordinate data, processing that data into a CSV format, and training a classification model to recognize different hand positions.

Technical Components
Hand Tracking: Real-time landmark detection via webcam.

Data Processing: Normalization and preparation of landmark coordinates.

Model Training: Implementation of a classification model to interpret gestures.

Built With
Python - Programming language.

MediaPipe - Framework for hand landmark detection.

OpenCV - Image processing and camera integration.

Scikit-learn - Machine learning model training.

NumPy/Pandas - Data manipulation and analysis.

Project Structure
Hand Tracker.py: Captures real-time hand movements and landmarks.

Data Processing.py: Formats captured data for the training pipeline.

Model Training.py: Trains the classifier using the processed dataset.

hand_landmarks_data.csv: Storage for landmark coordinate data.
