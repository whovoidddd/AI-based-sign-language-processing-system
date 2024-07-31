import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

dataset_path = r'C:\Users\vedant\Desktop\a to z new\SignImage256x256'

# Function to load images and labels
def load_data(dataset_path):
    images = []
    labels = []
    for folder in os.listdir(dataset_path):
        label = folder
        folder_path = os.path.join(dataset_path, folder)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
            img = cv2.resize(img, (256, 256))  # Resize to 256x256 pixels
            images.append(img)
            labels.append(label)
    images = np.array(images, dtype='float32') / 255.0  # Normalize images
    images = np.expand_dims(images, axis=-1)  # Add channel dimension
    labels = np.array(labels)
    return images, labels

# Load the dataset
images, labels = load_data(dataset_path)

# Convert labels to categorical (one-hot encoding)
label_to_id = {label: idx for idx, label in enumerate(np.unique(labels))}
id_to_label = {v: k for k, v in label_to_id.items()}
labels = np.array([label_to_id[label] for label in labels])
labels = tf.keras.utils.to_categorical(labels)

# Split dataset into training, validation, and test sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Define your CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_to_id), activation='softmax')  # Output layer with number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Evaluate the model on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# Save the model
model.save('sign_language_model_256.keras')
