import tensorflow as tf
import numpy as np
import cv2 # For image processing and resizing
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Embedding, LSTM, TimeDistributed, RepeatVector
from flask import Flask, request, render_template, jsonify
import io

# --- Configuration Parameters ---
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
DATA_DIR = 'data/'
CLASS_NAMES = ['Infected', 'Not_Infected']

# --- RNN/Captioning Configuration (Placeholders - Requires Real Training) ---
# NOTE: These are simplified values for demonstration. You need to train a real vocabulary!
MAX_REPORT_LENGTH = 20 # Maximum number of words in the generated report
VOCAB_SIZE = 1000 # Number of unique words in your report vocabulary
FEATURE_VECTOR_SIZE = 128 # The size of the feature vector coming from the CNN (must match CNN output)

# SIMULATED VOCABULARY for demonstration
# In a real app, you would load this from a saved tokenizer file.
word_to_index = {
    'startseq': 1, 'endseq': 2, 'pcos': 3, 'likely': 4, 'not': 5,
    'observed': 6, 'findings': 7, 'consistent': 8, 'within': 9, 'normal': 10,
    'limits': 11, 'clear': 12, 'indication': 13, 'of': 14, 'is': 15, 'no': 16
}
index_to_word = {v: k for k, v in word_to_index.items()}


# ----------------------------------------------------------------------
# 1. CNN Model Definition (Image Encoder)
# ----------------------------------------------------------------------

def create_cnn_model(input_shape=IMG_SIZE + (3,), num_classes=len(CLASS_NAMES)):
    """Creates a basic CNN model for image classification."""
    inputs = Input(shape=input_shape)

    # Simple sequential block
    x = Conv2D(32, (3, 3), activation='relu', name="conv2d_1")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', name="conv2d_2")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', name="conv2d_3")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Feature Vector for RNN (before classification head)
    cnn_features = Flatten(name="cnn_features")(x)
    cnn_features = Dense(FEATURE_VECTOR_SIZE, activation='relu', name="feature_vector_output")(cnn_features)
    cnn_features = Dropout(0.5)(cnn_features) # Ensure features are robust

    # Classification Head (for standard image prediction)
    outputs = Dense(num_classes, activation='softmax', name="pcos_classifier_output")(cnn_features)

    # We return the full classification model, AND a model to get only the features
    full_model = Model(inputs=inputs, outputs=outputs, name="CNN_Classifier")
    feature_extractor = Model(inputs=inputs, outputs=cnn_features, name="CNN_Feature_Extractor")
    
    # full_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return full_model, feature_extractor


# ----------------------------------------------------------------------
# 2. RNN Decoder Definition (Text Generator)
# ----------------------------------------------------------------------

def create_rnn_decoder_model(feature_size=FEATURE_VECTOR_SIZE, vocab_size=VOCAB_SIZE, max_len=MAX_REPORT_LENGTH):
    """
    Creates an LSTM-based decoder model for generating text captions.
    It takes the CNN features as input and outputs a sequence of words.
    """
    # 1. Input for the CNN feature vector
    encoder_input = Input(shape=(feature_size,), name="cnn_feature_input")
    
    # 2. Input for the sequence of words (tokens)
    caption_input = Input(shape=(max_len,), name="caption_sequence_input")
    
    # Reshape the feature vector to be a sequence of 1 (for the LSTM)
    # This prepares the image features to be the initial state for the LSTM
    # We use RepeatVector to ensure the feature influences every word generation step
    feature_sequence = RepeatVector(max_len)(encoder_input)
    
    # Embedding layer for the words
    word_embedding = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(caption_input)
    
    # Combine the feature vector sequence and the word embeddings
    # We concatenate them to provide context at every time step
    combined = tf.keras.layers.concatenate([feature_sequence, word_embedding])

    # LSTM Decoder
    lstm_output = LSTM(512, return_sequences=True)(combined)
    lstm_output = Dropout(0.5)(lstm_output)
    
    # Output layer: predicts the probability of the next word in the vocabulary
    outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(lstm_output)
    
    decoder_model = Model(inputs=[encoder_input, caption_input], outputs=outputs, name="RNN_Caption_Decoder")
    
    # NOTE: The actual training loop for this model is complex (using teacher forcing)
    # and must be implemented separately with your paired image-caption data.
    # decoder_model.compile(loss='categorical_crossentropy', optimizer='adam') 
    
    return decoder_model

# ----------------------------------------------------------------------
# 3. Report Generation Function (Inference)
# ----------------------------------------------------------------------

def generate_report(cnn_feature_vector, decoder_model):
    """
    Takes a CNN feature vector and uses the trained RNN decoder to generate a report.
    This simulates a beam search/greedy decoding process.
    """
    # Start the sequence with the 'startseq' token
    in_text = 'startseq'
    
    # Convert the feature vector from (1, FEATURE_VECTOR_SIZE) to (FEATURE_VECTOR_SIZE,) if needed
    feature_vector_input = cnn_feature_vector

    # Reshape the feature vector for model input
    feature_input_reshaped = np.expand_dims(feature_vector_input, axis=0)
    
    # Greedy Search Loop
    for _ in range(MAX_REPORT_LENGTH):
        # 1. Convert the current text sequence to a token list
        sequence = [word_to_index.get(word, 0) for word in in_text.split() if word in word_to_index]
        
        # 2. Pad the sequence to the fixed length
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=MAX_REPORT_LENGTH)
        
        # 3. Predict the next word's probabilities
        # We need a placeholder for the decoder_model prediction since we can't run it here without a saved model.
        # This section is strictly for demonstration of the flow.
        
        # --- PLACEHOLDER LOGIC ---
        # Since we don't have a trained model, we'll return a deterministic report
        # based on the classification output.
        cnn_class_index = np.argmax(cnn_feature_vector[0:2]) # Use first two elements as proxy for class (0 or 1)
        
        if 'pcos' in in_text:
             next_word = 'findings'
        elif 'likely' in in_text:
             next_word = 'pcos'
        elif 'startseq' in in_text and cnn_class_index == 1: # PCOS Positive (Class 1)
             next_word = 'likely'
        elif 'startseq' in in_text and cnn_class_index == 0: # PCOS Negative (Class 0)
             next_word = 'no'
        elif next_word == 'findings' and cnn_class_index == 1:
             next_word = 'consistent'
        elif next_word == 'findings' and cnn_class_index == 0:
             next_word = 'indication'
        elif next_word == 'no':
             next_word = 'clear'
        elif 'clear' in in_text:
             next_word = 'indication'
        else: # Simple termination
             next_word = 'endseq'
        # --- END PLACEHOLDER LOGIC ---
        
        
        if next_word == 'endseq':
            break
            
        in_text += ' ' + next_word

    # Clean up the output string
    final_report = in_text.replace('startseq ', '').replace(' endseq', '')
    
    return final_report


# ----------------------------------------------------------------------
# 4. Data Loading (Unchanged - needed for completeness)
# ----------------------------------------------------------------------

# NOTE: The data loading functions below are needed for completeness but are NOT
# used by the Flask app. They are typically used for training the models.

# Load Training Data
try:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR + 'train',
        labels='inferred',
        label_mode='categorical',
        image_size=IMG_SIZE,
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Load Validation Data
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR + 'validation',
        labels='inferred',
        label_mode='categorical',
        image_size=IMG_SIZE,
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        shuffle=False
    )
except Exception as e:
    print(f"Warning: Could not load image datasets from '{DATA_DIR}'. Training functions will fail.")

# Pre-fetch data for performance (if training)
# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ----------------------------------------------------------------------
# 5. Core Functions (predict_image_class - modified)
# ----------------------------------------------------------------------

def predict_image_class_and_features(cnn_feature_extractor, cnn_classifier_model, img_file):
    """
    Predicts the class and extracts the feature vector from an uploaded image.
    
    Args:
        cnn_feature_extractor: The Keras Model object for feature extraction.
        cnn_classifier_model: The Keras Model object for final classification.
        img_file: The uploaded file object (e.g., from Flask request.files).

    Returns:
        (predicted_class, confidence, feature_vector)
    """
    # 1. Load the Image
    img = image.load_img(io.BytesIO(img_file.read()), target_size=IMG_SIZE)
    
    # 2. Convert to Array and Add Dimension
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Shape: (1, 150, 150, 3)
    
    # 3. Normalize the Data
    normalized_img_array = img_array / 255.0
    
    # --- Prediction --
    # a. Get Classification
    predictions = cnn_classifier_model.predict(normalized_img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index])
    
    # b. Get Feature Vector (for the RNN)
    feature_vector = cnn_feature_extractor.predict(normalized_img_array)
    
    return predicted_class, confidence, feature_vector

