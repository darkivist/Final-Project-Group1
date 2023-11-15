#%%
# LOAD PACKAGES
import torch
import torch.nn as nn
#from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, GPT2LMHeadModel
#%%
# Loading Data
mwp = "insert here"
equation = "insert here"

#%%
#Initializing Tokenizer and Getting Inputs
checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(mwp,return_tensors="pt")
labels = tokenizer(equation, return_tensors="pt")
#%% 
# Model 
model = GPT2LMHeadModel.from_pretrained(checkpoint)
outputs = model(**inputs)

#%%
loss = outputs.loss
logits = outputs.logits

