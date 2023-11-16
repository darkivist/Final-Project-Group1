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
    
# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load your data
df = pd.read_csv('/home/ubuntu/NLP_Main/Final-Project-Group1/Code/carrie_sample_code.csv')
mwp = list(obj for obj in df["Problem"])
equation = list(obj for obj in df["Numeric Equation"])

# Split the data into training and validation sets
train_mwp, val_mwp, train_equation, val_equation = train_test_split(mwp, equation, test_size=0.1, random_state=42)

# Tokenize the data
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
train_encodings = tokenizer(train_mwp, truncation=True, padding=True)
val_encodings = tokenizer(val_mwp, truncation=True, padding=True)

# Create datasets
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=None,  # Pass None because we provide the tokenized data directly
    text_files=train_encodings["input_ids"],
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    ),
)
val_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=None,  # Pass None because we provide the tokenized data directly
    text_files=val_encodings["input_ids"],
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    ),
)

# Fine-tune the GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
training_args = TrainingArguments(
    output_dir="./gpt2_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=train_dataset.data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# %%
