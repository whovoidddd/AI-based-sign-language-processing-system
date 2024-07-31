import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time

# Load the pre-trained models
model_a_to_z = tf.keras.models.load_model('sign_language_model_256.keras')
model_space_next = tf.keras.models.load_model('sign_language_space_next_model.keras')

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to preprocess the image for model prediction
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (256, 256))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Label dictionaries
label_dict_a_to_z = {i: chr(97 + i) for i in range(26)}  # 'a' to 'z'
label_dict_space_next = {0: 'space', 1: 'next'}

# Start webcam
cap = cv2.VideoCapture(0)

# Timer
last_prediction_time = time.time()
prediction_interval = 5  # seconds
predicted_label = ""

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get the bounding box of the hand
                h, w, c = frame.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x, x_min), min(y, y_min)
                    x_max, y_max = max(x, x_max), max(y, y_max)
                
                # Add some margin to the bounding box
                margin = 20
                x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
                x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)
                
                # Extract the ROI
                roi = frame[y_min:y_max, x_min:x_max]
                
                # Predict every 5 seconds
                current_time = time.time()
                if current_time - last_prediction_time >= prediction_interval:
                    processed_roi = preprocess_image(roi)
                    
                    # Predict using both models
                    prediction_a_to_z = model_a_to_z.predict(processed_roi)[0]
                    prediction_space_next = model_space_next.predict(processed_roi)[0]
                    
                    # Determine the final label based on highest confidence score
                    max_prob_a_to_z = np.max(prediction_a_to_z)
                    max_prob_space_next = np.max(prediction_space_next)
                    
                    if max_prob_a_to_z > max_prob_space_next:
                        predicted_label = label_dict_a_to_z[np.argmax(prediction_a_to_z)]
                    else:
                        predicted_label = label_dict_space_next[np.argmax(prediction_space_next)]
                    
                    last_prediction_time = current_time

        # Display the label in the top-left corner
        cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Sign Language Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
