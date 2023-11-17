#%%
# LOAD PACKAGES
import pandas as pd
import torch
import torch.nn as nn
#from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
# Loading Data
df = pd.read_csv('/home/ubuntu/NLP_Main/Final-Project-Group1/Code/cleaned_dataset.csv')
mwp = df["Question"]
mwp = list(obj for obj in mwp) #need to make pandas object into list of strings

equation = df["Equation"]
equation = list(obj for obj in equation)
print(len(equation))
print(len(mwp))
#%%
#Initializing Tokenizer and Getting Inputs
checkpoint = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

max_sequence_length = max(len(mwp), len(equation))

inputs = tokenizer(mwp, padding="max_length", truncation=True, max_length=max_sequence_length, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}
labels = tokenizer(equation, padding="max_length", truncation=True, max_length=max_sequence_length, return_tensors="pt")
labels = {key: value.to(device) for key, value in labels.items()}
print(inputs["input_ids"].shape)
print(labels["input_ids"].shape)

#%% 
# Model 
model = GPT2LMHeadModel.from_pretrained(checkpoint).to(device)
#%%
#Training
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
#%%

# %%
n_epochs = 5

for epoch in range(n_epochs):
        # Forward pass
        outputs = model(**inputs, labels=labels["input_ids"])
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss for each epoch
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")
#%%
test_prompt = "David has 48 marbles. He puts them into 4 bags. How many marbles are there in each bag?"
test_input = tokenizer(test_prompt, padding="max_length", truncation=True, max_length=max_sequence_length, return_tensors="pt")
# Tokenize the test prompt

output_ids = model.generate(test_input["input_ids"], max_length=50, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
generated_equation = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print the generated equation
print("Input Math Word Problem:")
print(test_prompt)
print("\nGenerated Equation:")
print(generated_equation)
#%%    
#### OTHER MODEL
import pandas as pd
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.nn import CrossEntropyLoss

# Load the data
df = pd.read_csv('/home/ubuntu/NLP_Main/Final-Project-Group1/Code/cleaned_dataset.csv')

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
# Reduce batch size
batch_size = 4  # Adjust according to your system's capacity
tokenized_inputs = tokenizer(preprocessed_input_exps, return_tensors="pt", padding=True, truncation=True, max_length=64, max_batch_size=batch_size)
tokenized_targets_no_variable = tokenizer(preprocessed_target_exps, return_tensors="pt", padding=True, truncation=True, max_length=64, max_batch_size=batch_size)

#tokenized_inputs = tokenizer(preprocessed_input_exps, return_tensors="pt", padding=True, truncation=True, max_length=64)
#tokenized_targets_no_variable = tokenizer(preprocessed_target_exps, return_tensors="pt", padding=True, truncation=True, max_length=64)

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


# %%
