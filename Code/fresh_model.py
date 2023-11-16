#%%
import pandas as pd
import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import torch.optim as optim
import torch.nn.functional as F
#%%
# Load the data
df = pd.read_csv('/home/ubuntu/NLP_Main/Final-Project-Group1/Code/carrie_sample_code.csv')

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
#%%
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Tokenize text
tokenizer = T5Tokenizer.from_pretrained("t5-base")
tokenized_inputs = tokenizer(preprocessed_input_exps, return_tensors="pt", padding=True, truncation=True, max_length=64)
tokenized_targets = tokenizer(preprocessed_target_exps, return_tensors="pt", padding=True, truncation=True, max_length=64)

# Create Seq2Seq model
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    attention_mask = tokenized_inputs["attention_mask"]
    outputs = model(input_ids=tokenized_inputs["input_ids"], attention_mask=attention_mask, labels=tokenized_targets["input_ids"])

    loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), tokenized_targets["input_ids"].view(-1))

    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
#%%



# %%
