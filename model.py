import tensorflow as tf
import numpy as np
import cv2 # For image processing and resizing (You may need to run: pip install opencv-python)
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, jsonify
import io

# --- Configuration Parameters ---
# All images will be resized to this size (e.g., 150x150 pixels)
IMG_SIZE = (150, 150)
# Batch size for training
BATCH_SIZE = 32
# Directory where your data folders (train, test, validation) are located
DATA_DIR = 'data/' 

# --- Load Training Data ---
# This function automatically infers labels from the subfolder names ('infected', 'not_infected')
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR + 'train',
    labels='inferred',#The image_dataset_from_directory function looks at the name of the subfolders inside the train directory to assign the correct label to every image. This is a crucial step in preparing the training data.
    label_mode='categorical', # Use 'categorical' for one-hot encoded labels (e.g., [1, 0] or [0, 1])
    image_size=IMG_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=True # Shuffle training data for better learning
)

# --- Load Validation Data ---
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR + 'validation',
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# --- Load Test Data ---
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR + 'test',
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Display the class names and number of batches
print(f"\nClass Names: {train_ds.class_names}") 
print(f"Number of training images: {len(train_ds) * BATCH_SIZE}") # Approximation

# --- Rescaling (Normalization) ---
# Create a layer to scale pixel values from [0, 255] to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1./255)  #Normalized Value = raw pixel value/255

# Apply the normalization layer to the datasets
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# --- Optimization ---
# Configure the datasets for faster loading during training
AUTOTUNE = tf.data.AUTOTUNE #It tells TensorFlow to automatically figure out the best way to handle parallel tasks (like loading images, normalizing, and augmenting).
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

# Assuming IMG_SIZE has been defined (e.g., (150, 150))

# --- 1. Define the Augmentation Pipeline ---
# This layer introduces variability to the small dataset during training
data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"), 
    RandomRotation(0.2),                   
    RandomZoom(0.2)                        
])

# --- 2. Model Definition with Augmentation and Dropout ---
cnn_model_robust = Sequential([
  # Layer 0: Augmentation - ONLY applied during training
  data_augmentation, 
  
  # 1st Conv Block
  Conv2D(32, 3, activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
  MaxPooling2D(),
  
  # 2nd Conv Block
  Conv2D(64, 3, activation='relu'),
  MaxPooling2D(),
  
  # Classification Block
  Flatten(),
  
  # Dense Layer with Dropout (Regularization)
  Dense(128, activation='relu'),
  Dropout(0.5), # Ignores 50% of neurons randomly to force generalization
  
  # Output Layer
  Dense(2, activation='softmax') # 2 classes: infected and not_infected
])

# The primary job of the softmax activation function, 
# when used in the output layer of your CNN, is to turn raw, arbitrary scores (called logits) into a probability distribution.

# --- 3. Compile and Train (Example) ---
# Use this new model object (cnn_model_robust) for compiling and fitting
cnn_model_robust.compile(
    # optimizer='adam',An optimizer is used in a Convolutional Neural Network (CNN) to minimize the model's prediction error 
    # by iteratively adjusting its internal parameters (weights and biases) during training.
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nStarting training with loaded datasets...")
history = cnn_model_robust.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15 # Train for a few more epochs since you have very little data
)

# --- Evaluate ---
loss, accuracy = cnn_model_robust.evaluate(test_ds)
print(f"\nTest set accuracy: {accuracy:.4f}")
cnn_model_robust.save('pcos_cnn_model.keras') 
print("\nModel successfully saved as pcos_cnn_model.keras")

# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing import image
# tf.keras.models.load_model('pcos_cnn_model.keras')

# cnn_model_robust = tf.keras.models.load_model('pcos_cnn_model.keras')

# # --- Configuration (Must match your training setup) ---
# IMG_SIZE = (150, 150)
# CLASS_NAMES = ['Infected', 'Not_Infected'] # Matches your folder names

# # --- Load the new image ---
# def preprocess_and_predict(model, img_path):
#     # 1. Load and Resize Image
#     # target_size ensures the image is resized to (150, 150)
#     img = image.load_img(img_path, target_size=IMG_SIZE)
    
#     # 2. Convert to Array and Add Batch Dimension
#     # The array shape becomes (150, 150, 3)
#     img_array = image.img_to_array(img)
    
#     # The shape must be (1, 150, 150, 3) for the model.predict method
#     img_array = np.expand_dims(img_array, axis=0) 
    
#     # 3. Normalize the Data (Scale to [0, 1])
#     # The model was trained on normalized data!
#     normalized_img_array = img_array / 255.0
    
#     # --- Prediction ---
#     # The model outputs a probability vector (e.g., [0.1, 0.9])
#     predictions = model.predict(normalized_img_array)
    
#     # Find the index of the highest probability
#     predicted_class_index = np.argmax(predictions[0])
    
#     # Get the class name and confidence
#     predicted_class = CLASS_NAMES[predicted_class_index]
#     confidence = predictions[0][predicted_class_index]
    
#     return predicted_class, confidence

# # --- Example Usage ---
# # Ensure your model object (cnn_model_robust) is loaded or defined and trained
# # Example: cnn_model_robust = tf.keras.models.load_model('path/to/your/saved_model')
# new_image_path = r"C:\Users\Karthickraja.S\Downloads\download (2).jpeg" 

# # Assuming your trained model object is named cnn_model_robust
# predicted_label, confidence_score = preprocess_and_predict(cnn_model_robust, new_image_path)

# print(f"\n--- Prediction Result ---")
# print(f"Image: {new_image_path}")
# print(f"Predicted Class: **{predicted_label}**")
# print(f"Confidence: **{confidence_score * 100:.2f}%**")