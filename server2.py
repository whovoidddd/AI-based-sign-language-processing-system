from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variable
apiKey = os.getenv('API_KEY')

app = Flask(__name__)

# Load your trained models
model_letters = tf.keras.models.load_model('sign_language_model_256.keras')
model_special = tf.keras.models.load_model('sign_language_space_next_model.keras')

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # Import drawing utilities

# Initialize webcam
cap = None  # Will be initialized later

word_formed = ""
input_letter = ""
last_prediction_time = time.time()
final_prediction = ""  # To store the final prediction to be displayed in the input
last_detected_time = time.time()
space_predicted = False  # Track if space was predicted
next_predicted_after_space = False  # Track if 'next' was predicted after 'space'

def preprocess_frame(frame):
    # Convert to RGB as MediaPipe Hands model expects RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Use MediaPipe to detect landmarks
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Find bounding box around hand landmarks
            landmarks = hand_landmarks.landmark
            x_min, y_min = frame.shape[1], frame.shape[0]
            x_max, y_max = 0, 0
            for landmark in landmarks:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Ensure the extracted region is within bounds
            if 0 <= y_min < y_max <= frame.shape[0] and 0 <= x_min < x_max <= frame.shape[1]:
                # Extract ROI from the frame
                roi_frame = frame[max(0, y_min - 20):y_max + 20, max(0, x_min - 20):x_max + 20]
                
                # Resize ROI to match model's expected sizing (256x256)
                resized_roi = cv2.resize(roi_frame, (256, 256))
                
                # Convert to grayscale for letters model
                gray_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
                
                # Normalize the ROI
                normalized_roi_letters = gray_roi.astype('float32') / 255.0
                normalized_roi_special = gray_roi.astype('float32') / 255.0  # Convert to grayscale
                
                # Expand dimensions to create a batch of 1
                processed_roi_letters = np.expand_dims(normalized_roi_letters, axis=-1)
                processed_roi_letters = np.expand_dims(processed_roi_letters, axis=0)  # Add batch dimension
                processed_roi_special = np.expand_dims(normalized_roi_special, axis=-1)
                processed_roi_special = np.expand_dims(processed_roi_special, axis=0)  # Add batch dimension
                
                # Determine which hand is detected
                hand_label = 'unknown'
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Check the handedness
                        handedness = results.multi_handedness[results.multi_hand_landmarks.index(hand_landmarks)]
                        hand_label = handedness.classification[0].label  # 'Left' or 'Right'

                return processed_roi_letters, processed_roi_special, frame, hand_label
    
    return None, None, frame, 'unknown'

def generate_frames():
    global cap, word_formed, input_letter, last_prediction_time, final_prediction, last_detected_time, space_predicted, next_predicted_after_space

    while True:
        if cap is None:
            continue

        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret:
            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Preprocess frame to extract ROI and draw landmarks
            processed_roi_letters, processed_roi_special, frame_with_landmarks, hand_label = preprocess_frame(frame)
            
            if processed_roi_letters is not None and processed_roi_special is not None:
                current_time = time.time()
                time_elapsed = current_time - last_prediction_time
                last_detected_time = current_time
                
                if time_elapsed > 3.0:
                    # Perform prediction on the processed ROI based on detected hand
                    if hand_label == 'Right':
                        prediction = model_letters.predict(processed_roi_letters)
                        final_prediction = chr(np.argmax(prediction) + ord('a'))
                    elif hand_label == 'Left':
                        prediction = model_special.predict(processed_roi_special)
                        final_prediction = 'space' if np.argmax(prediction) == 0 else 'next'
                    else:
                        final_prediction = ""  # Unknown hand

                    print(f"Prediction: {prediction}, Hand: {hand_label}, Final Prediction: {final_prediction}")
                    
                    # Update the word formed based on 'next', 'space', or letter
                    if final_prediction == 'next':
                        if next_predicted_after_space:
                            word_formed += " "  # Append space only if 'next' is predicted after 'space'
                        word_formed += input_letter
                        input_letter = ""
                        space_predicted = False
                        next_predicted_after_space = False
                    elif final_prediction == 'space':
                        space_predicted = True
                    else:
                        input_letter = final_prediction

                    # Update last prediction time
                    last_prediction_time = current_time

            # Display the input letter on the frame
            cv2.putText(frame_with_landmarks, f'Input: {final_prediction}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the word formed on the frame with appropriate spacing
            formatted_word_formed = word_formed.replace(" ", " ")
            cv2.putText(frame_with_landmarks, f'Word formed: {formatted_word_formed}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Check if no hand has been detected for 5 seconds and update the final word
            if time.time() - last_detected_time > 5.0:
                final_word_display = word_formed
                app.config['final_word_display'] = final_word_display

            # Encode frame as JPEG image
            ret, buffer = cv2.imencode('.jpg', frame_with_landmarks)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/main')
def main():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
    return jsonify({'message': 'Detection started'})

@app.route('/get_final_word')
def get_final_word():
    final_word_display = app.config.get('final_word_display', '')
    return jsonify({'final_word': final_word_display})

@app.route('/generate_image', methods=['POST'])
def generate_image():
    prompt = request.json.get('prompt', '')

    requestBody = {
        'prompt': prompt,
        'image_size': 'square_hd',
        'num_inference_steps': 4,
        'num_images': 1,
        'format': 'jpeg'
    }

    try:
        response = requests.post('https://fal.run/fal-ai/fast-lightning-sdxl',
                                 headers={'Authorization': f'Key {apiKey}', 'Content-Type': 'application/json'},
                                 json=requestBody)
        response.raise_for_status()
        data = response.json()
        imageUrl = data['images'][0]['url']
        return jsonify({'image_url': imageUrl})

    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
