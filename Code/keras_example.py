#%%
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Flatten, RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
# %%

# Example dataset (math word problems and their corresponding numeric equations)
math_problems = ['What is 2 + 3?', 'If John has 5 apples and gives 2 to Mary, how many apples does John have now?']
numeric_equations = ['2 + 3', '5 - 2']

# Tokenize input and output sequences
t_in = Tokenizer()
t_in.fit_on_texts(math_problems)
encoded_in = t_in.texts_to_sequences(math_problems)
vocab_size_in = len(t_in.word_index) + 1

t_out = Tokenizer()
t_out.fit_on_texts(numeric_equations)
encoded_out = t_out.texts_to_sequences(numeric_equations)
vocab_size_out = len(t_out.word_index) + 1

# Pad sequences to a fixed length
max_len_input = max(len(seq) for seq in encoded_in)
max_len_output = max(len(seq) for seq in encoded_out)

padded_input_sequences = pad_sequences(encoded_in, maxlen=max_len_input, padding='post')
padded_output_sequences = pad_sequences(encoded_out, maxlen=max_len_output, padding='post')

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    padded_input_sequences, padded_output_sequences, test_size=0.2, random_state=42
)


# Define the Seq2Seq model
model = Sequential()
model.add(Embedding(input_dim=vocab_size_in, output_dim=50, input_length=max_len_input))
model.add(LSTM(100))
model.add(RepeatVector(max_len_output))  # Repeat the context vector to match output sequence length
model.add(LSTM(100, return_sequences=True))  # Decoder LSTM layer
model.add(Dense(vocab_size_out, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with validation data
model.fit(X_train, y_train, epochs=30, batch_size=1, validation_data=(X_val, y_val))
# %%
# Test example
test_math_problem = ['If there are 3 apples and you take away 1, how many apples do you have?']
test_encoded_input = t_in.texts_to_sequences(test_math_problem)
padded_test_input = pad_sequences(test_encoded_input, maxlen=max_len_input, padding='post')

# Make predictions
predicted_output_sequence = model.predict(padded_test_input)

# Decode the predicted output sequence
predicted_numeric_equation = [t_out.index_word[idx] for idx in predicted_output_sequence.argmax(axis=-1)[0] if idx != 0]
predicted_numeric_equation = ' '.join(predicted_numeric_equation)

print("Input Math Problem:", test_math_problem[0])
print("Predicted Numeric Equation:", predicted_numeric_equation)

# %%
# Output wasn't accurate but need more training data.
# I just wanted get comfortable with writing a model for our project. 