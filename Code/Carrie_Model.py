#%%
# LOAD PACKAGES
import pandas as pd
import torch
import torch.nn as nn
#from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch.optim as optim
from torch.utils.data import DataLoader
#%%
# Loading Data
df = pd.read_csv('/home/ubuntu/NLP_Main/Final-Project-Group1/Code/carrie_sample_code.csv')
mwp = df["Problem"]
mwp = list(obj for obj in mwp) #need to make pandas object into list of strings

equation = df["Numeric Equation"]
equation = list(obj for obj in equation)
print(len(equation))
print(len(mwp))

#%%
#Initializing Tokenizer and Getting Inputs
checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer(mwp, padding=True, truncation=True,return_tensors="pt")
labels = tokenizer(equation, padding=True, truncation=True,  return_tensors="pt")
#%% 
# Model 
model = GPT2LMHeadModel.from_pretrained(checkpoint)
#%%
#Training
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# %%
n_epochs = 10

for epoch in range(n_epochs):
    # Forward pass
    outputs = model(**inputs, labels=labels["input_ids"])
    loss = outputs.loss

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss for each epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
#%%    
#### OTHER MODEL
import pandas as pd
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.nn import CrossEntropyLoss

# Load the data
df = pd.read_csv('/home/ubuntu/NLP_Main/Final-Project-Group1/cleaned_dataset.csv')

# Extract input and target expressions
input_exps = list(df['Question'].values)
target_exps = list(df['Equation'].values)

# Preprocess input and target expressions
def preprocess_input(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,’])", r" \1 ", sentence)
    sentence = re.sub(r"([0-9])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = sentence.rstrip().strip()
    return sentence

def preprocess_target(sentence):
    sentence = sentence.lower().strip()
    return sentence

preprocessed_input_exps = list(map(preprocess_input, input_exps))
preprocessed_target_exps = list(map(preprocess_target, target_exps))


# Tokenize text
tokenizer = T5Tokenizer.from_pretrained("t5-small")
tokenized_inputs = tokenizer(preprocessed_input_exps, return_tensors="pt", padding=True, truncation=True, max_length=64)
tokenized_targets_no_variable = tokenizer(preprocessed_target_exps, return_tensors="pt", padding=True, truncation=True, max_length=64)

# Create Seq2Seq model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss()

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    attention_mask = tokenized_inputs["attention_mask"]
    outputs = model(input_ids=tokenized_inputs["input_ids"], attention_mask=attention_mask, labels=tokenized_targets_no_variable["input_ids"])

    loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), tokenized_targets_no_variable["input_ids"].view(-1))

    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

def generate_prediction(model, tokenizer, input_text):
    model.eval()  # Set the model to evaluation mode
    input_text = preprocess_input(input_text)
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=64)["input_ids"]
    
    # Generate output from the model
    output_ids = model.generate(input_ids)
    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return decoded_output
# Replace "Your test sentence goes here." with your actual test sentence
test_sentence = "David has 48 marbles. He puts them into 4 bags. How many marbles are there in each bag?"

# Generate prediction
with torch.no_grad():  # No need for gradient computation during inference
    prediction = generate_prediction(model, tokenizer, test_sentence)

print(f"Test Sentence: {test_sentence}")
print(f"Generated Prediction: {prediction}")

