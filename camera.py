import cv2
import numpy as np
from keras.models import load_model
from utils import WebcamVideoStream

# Load the pre-trained emotion recognition model
emotion_model = load_model(r'C:\Users\Dell\Downloads\Emotion-Music-Recommendation\best_efficientnet_model.keras')

# Emotion labels as per the FER2013 dataset
# 😄 Happy, 😠 Angry, 😮 Surprised, 😞 Sad
emotion_labels = ['Angry', 'Happy', 'Sad', 'Surprised']

# Function to detect emotion from webcam feed
def detect_emotion():
    # Initialize the webcam video stream
    video_stream = WebcamVideoStream(src=0).start()
    
    # Load face cascade classifier once
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Pre-allocate arrays for better performance
    frame_buffer = np.zeros((1, 96, 96, 3), dtype=np.float32)
    
    while True:
        frame = video_stream.read()
        if frame is None:
            continue

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,  # Increased from 1.3 for faster detection
            minNeighbors=4,   # Reduced from 5 for faster detection
            minSize=(30, 30)  # Added minimum size for faster processing
        )

        for (x, y, w, h) in faces:
            # Extract and preprocess face region
            roi_gray = gray_frame[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (96, 96))
            
            # Convert grayscale to RGB (3 channels)
            roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
            
            # Normalize and reshape
            frame_buffer[0] = roi_rgb / 255.0

            # Predict emotion
            prediction = emotion_model.predict(frame_buffer, verbose=0)  # Disabled verbose output
            max_index = int(np.argmax(prediction))
            predicted_emotion = emotion_labels[max_index]

            # Draw rectangle and emotion label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, predicted_emotion, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Display the frame
            cv2.imshow('Emotion Detection', frame)

            # Stop the stream and return detected emotion
            video_stream.stop()
            cv2.destroyAllWindows()
            return predicted_emotion

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_stream.stop()
    cv2.destroyAllWindows()
    return None

