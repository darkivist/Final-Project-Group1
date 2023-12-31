from pathlib import Path
import torch
import torch
import torch.nn as nn
from config import get_config, latest_weights
from train import get_model, get_ds, run_validation
from Run_Transformer import translate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
config = get_config()
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

# Load the pretrained weights
model_filename = latest_weights(config)
print(model_filename)
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])

#run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: print(msg), 0, None, num_example=1)

print(translate("I have 10 Apples. How many apples do I have ?"))
