import cv2
import os
import numpy as np
import mediapipe as mp

directory = 'SignImage256x256/'

if not os.path.exists(directory):
    os.makedirs(directory)
for letter in range(ord('a'), ord('z') + 1):
    letter_folder = os.path.join(directory, chr(letter))
    if not os.path.exists(letter_folder):
        os.makedirs(letter_folder)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  

# Open the webcam
cap = cv2.VideoCapture(0)

# Define constants
ROI_SIZE = (300, 300)  # Size of the ROI rectangle
MAX_IMAGES_PER_LABEL = 500  # Maximum number of images to capture per label

# Initialize variables
current_label = None
captured_images = {chr(key): 0 for key in range(ord('a'), ord('z') + 1)}  # Dictionary to count images per letter

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip the frame horizontally for a mirror-like effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for processing with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract ROI around hand landmarks for saving
                hand_roi = np.zeros_like(frame)  # Create black background
                mp_drawing.draw_landmarks(hand_roi, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Find bounding box around hand landmarks
                landmarks = hand_landmarks.landmark
                x_min, y_min = frame.shape[1], frame.shape[0]
                x_max, y_max = 0, 0
                for landmark in landmarks:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    if x < x_min:
                        x_min = x
                    if y < y_min:
                        y_min = y
                    if x > x_max:
                        x_max = x
                    if y > y_max:
                        y_max = y

                # Crop the region around hand landmarks from the original frame
                roi_frame = frame[max(0, y_min - 20):y_max + 20,
                                  max(0, x_min - 20):x_max + 20]

                # Resize ROI to 256x256
                roi_frame = cv2.resize(roi_frame, (256, 256))

                # Process frame (grayscale)
                processed_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    # Save image to 'q' folder if not exceeded max limit
                    if captured_images['q'] < MAX_IMAGES_PER_LABEL:
                        filename = f"{directory}q/q_{captured_images['q']}.jpg"
                        cv2.imwrite(filename, processed_frame)
                        print(f"Saved {filename}")
                        captured_images['q'] += 1
                elif key >= ord('a') and key <= ord('z'):
                    letter = chr(key)

                    # Reset if new label is pressed
                    if letter != current_label:
                        current_label = letter
                        captured_images[current_label] = 0

                    # Save image to corresponding folder if not exceeded max limit
                    if captured_images[current_label] < MAX_IMAGES_PER_LABEL:
                        filename = f"{directory}{current_label}/{current_label}_{captured_images[current_label]}.jpg"
                        cv2.imwrite(filename, processed_frame)
                        print(f"Saved {filename}")
                        captured_images[current_label] += 1

        # Display the frame with landmarks
        cv2.imshow('Hand Landmarks Detection', frame)

        # Check if user wants to exit
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
