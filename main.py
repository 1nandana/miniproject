import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained emotion detection model
model = load_model('model.h5')

# Define emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Define function to detect emotion in a given frame
def detect_emotion(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize the frame to 48x48 pixels
    resized = cv2.resize(gray, (48, 48))
    # Normalize the pixel values between 0 and 1
    normalized = resized / 255.0
    # Reshape the image to match the input shape of the model
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    # Make a prediction using the model
    prediction = model.predict(reshaped)
    # Get the index of the highest confidence emotion
    index = np.argmax(prediction)
    # Get the corresponding emotion label
    emotion = EMOTIONS[index]
    return emotion

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Loop over frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    # Flip the frame horizontally (optional)
    frame = cv2.flip(frame, 1)
    # Detect emotion in the frame
    emotion = detect_emotion(frame)
    # Display the emotion label on the frame
    cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Display the frame in a window
    cv2.imshow('Emotion Detection', frame)
    # Wait for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
