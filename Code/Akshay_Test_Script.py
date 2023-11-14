import pandas as pd
import numpy as np
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

train = "7 red apples and 2 green apples are in the basket. how many apples are in the basket ?"
label = "+72"

## Preprocessing
input_text = train.lower().split()  # Convert to lowercase and split the input text into words
output_label = label

word_to_index = {word: i for i, word in enumerate(set(input_text))}
print(word_to_index)
index_to_word = {i: word for i, word in enumerate(set(input_text))}
print(index_to_word)


input_sequence = [word_to_index[word] for word in input_text]
print(input_sequence)
output_sequence = np.array([float(output_label)])
print(output_sequence)

# Contruction LTSM
model = Sequential()
model.add(Embedding(input_dim=len(word_to_index), output_dim=50, input_length=len(input_sequence)))
model.add(LSTM(100))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(np.array([input_sequence]), output_sequence, epochs=10, verbose=2)

# Evaluate the model
loss = model.evaluate(np.array([input_sequence]), output_sequence)
print(f'Loss: {loss}')

# Make predictions
prediction = model.predict(np.array([input_sequence]))
print(f'Prediction: {prediction[0][0]}')