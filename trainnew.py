import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Constants
IMAGE_SIZE = 256  # Change to 256x256
NUM_CLASSES = 2  # 'space' and 'next'
NUM_IMAGES_PER_CLASS = 500
DATA_DIR = r'C:\Users\vedant\Desktop\a to z new\CustomSignImages'

# Function to load and preprocess images
def load_images(data_dir):
    images = []
    labels = []
    label_dict = {'space': 0, 'next': 1}

    for label in label_dict.keys():
        label_dir = os.path.join(data_dir, label)
        for image_file in os.listdir(label_dir)[:NUM_IMAGES_PER_CLASS]:
            image_path = os.path.join(label_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            images.append(image)
            labels.append(label_dict[label])

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Load the dataset
images, labels = load_images(DATA_DIR)

# Normalize the images
images = images.astype('float32') / 255.0

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to categorical format
y_train = to_categorical(y_train, NUM_CLASSES)
y_val = to_categorical(y_val, NUM_CLASSES)

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train[..., np.newaxis], y_train, epochs=15, batch_size=32, validation_data=(X_val[..., np.newaxis], y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val[..., np.newaxis], y_val)
print(f'Validation accuracy: {accuracy * 100:.2f}%')

# Save the model
model.save('sign_language_space_next_model.keras')
