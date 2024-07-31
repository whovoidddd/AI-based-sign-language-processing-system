import cv2
import os
import numpy as np
import mediapipe as mp


directory = 'CustomSignImages/'

labels = ['space', 'next']
for label in labels:
    label_folder = os.path.join(directory, label)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks

# Open the webcam
cap = cv2.VideoCapture(0)

# Define constants
ROI_SIZE = (300, 300)  # Size of the ROI rectangle
MAX_IMAGES_PER_LABEL = 2000  # Maximum number of images to capture per label

# Initialize variables
captured_images = {label: 0 for label in labels}  # Dictionary to count images per label

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
                if key == ord(' '):  # Space key for 'space' label
                    label = 'space'
                    if captured_images[label] < MAX_IMAGES_PER_LABEL:
                        filename = f"{directory}{label}/{label}_{captured_images[label]}.jpg"
                        cv2.imwrite(filename, processed_frame)
                        print(f"Saved {filename}")
                        captured_images[label] += 1
                elif key == ord('n'):  # 'n' key for 'next' label
                    label = 'next'
                    if captured_images[label] < MAX_IMAGES_PER_LABEL:
                        filename = f"{directory}{label}/{label}_{captured_images[label]}.jpg"
                        cv2.imwrite(filename, processed_frame)
                        print(f"Saved {filename}")
                        captured_images[label] += 1

        # Display the frame with landmarks
        cv2.imshow('Hand Landmarks Detection', frame)

        # Check if user wants to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
