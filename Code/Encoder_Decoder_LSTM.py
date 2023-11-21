import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention

df = pd.read_csv('SVAMP_CSV.csv')

df['word_problems'] = df['Body'] + df['Question']
# Preprocess the data
input_texts = df['word_problems'].values
target_texts = df['Equation'].values

tokenizer_input = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer_input.fit_on_texts(input_texts)
input_sequences = tokenizer_input.texts_to_sequences(input_texts)
input_sequences = pad_sequences(input_sequences)

tokenizer_target = Tokenizer(filters='', lower=False)
tokenizer_target.fit_on_texts(target_texts)
target_sequences = tokenizer_target.texts_to_sequences(target_texts)
target_sequences = pad_sequences(target_sequences)

# Define model parameters
vocab_size_input = len(tokenizer_input.word_index) + 1
vocab_size_target = len(tokenizer_target.word_index) + 1
latent_dim = 256

# Build the model
# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(vocab_size_input, latent_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(vocab_size_target, latent_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

# Attention mechanism
attention_layer = Attention()([decoder_outputs, encoder_outputs])
context_vector = attention_layer[:, :, :latent_dim]

# Concatenate decoder_outputs and context_vector
decoder_combined_context = tf.concat([decoder_outputs, context_vector], axis=-1)

# Dense layer for output
decoder_dense = Dense(vocab_size_target, activation='softmax')
output = decoder_dense(decoder_combined_context)

# Define the model
model = Model([encoder_inputs, decoder_inputs], output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

epochs = 50
batch_size = 64

# Train the model
model.fit([input_sequences, target_sequences[:, :-1]], target_sequences[:, 1:], epochs=epochs, batch_size=batch_size, validation_split=0.2)
'''
 target_sequences[:, :-1]], target_sequences[:, 1:] : is called teacher forcing. 
'''
model.summary()



example_index = np.random.randint(0, len(input_sequences))

# Get the input sequence
input_sequence = input_sequences[example_index : example_index + 1]

# Get the target sequence (real output)
target_sequence = target_sequences[example_index : example_index + 1]

# Generate the predicted output
predicted_output = model.predict([input_sequence, target_sequence[:, :-1]])

# Convert sequences back to text
input_text = tokenizer_input.sequences_to_texts(input_sequence)[0]
real_output_text = tokenizer_target.sequences_to_texts(target_sequence)[0]
predicted_output_text = tokenizer_target.sequences_to_texts(np.argmax(predicted_output, axis=-1))[0]

# Print the example
print("Input: ", input_text)
print("Real Output: ", real_output_text)
print("Predicted Output: ", predicted_output_text)
