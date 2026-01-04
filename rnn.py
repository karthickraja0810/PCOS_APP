import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding

# --- 1. Data Preparation ---

# Configuration
max_features = 10000  # Consider only the top 10,000 most frequent words
maxlen = 500          # Max length of a review (truncate longer reviews)

# Load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# --- 2. Model Definition ---

# Initialize the model
model = Sequential()

# Add layers
model.add(Embedding(max_features, 32)) 
model.add(SimpleRNN(32))              
model.add(Dense(1, activation='sigmoid')) 

# --- 3. Compile and Train ---

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model (fit to data)
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=64,
                    validation_data=(x_test, y_test))

# Evaluate the model
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Test Accuracy: {acc}')