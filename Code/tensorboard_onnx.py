#note - Adam, 1e-5, batch size 16, 200 epochs produced 4 correct answers from first 5 records in val set
#use train.csv with 9000 records

from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.utils.data import Dataset
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import torch.onnx
import netron

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up TensorBoard writer
writer = SummaryWriter(log_dir=training_args.logging_dir)

# Training Loop
for epoch in range(training_args.num_train_epochs):
    # Training
    trainer.train()

    # Save model checkpoint
    if (epoch + 1) % training_args.save_steps == 0:
        model_checkpoint_path = f'./results/checkpoint-{epoch+1}'
        trainer.save_model(model_checkpoint_path)

    # Log histograms of model weights
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch + 1)

    # Check if the current epoch equals the maximum number of epochs
    if epoch + 1 == training_args.num_train_epochs:
        break  # Exit the training loop after completing the specified epochs

# Ensure the model is in evaluation mode
model.eval()

# Dummy input data
max_length = 128
dummy_input = {
    "input_ids": torch.zeros([1, max_length], dtype=torch.long).to(device),
    "attention_mask": torch.ones([1, max_length], dtype=torch.long).to(device),
    "equation_ids": torch.ones([1, max_length], dtype=torch.long).to(device),
    "decoder_input_ids": torch.zeros([1, max_length], dtype=torch.long).to(device)  # Add decoder_input_ids
}

# Export the model to ONNX format
torch.onnx.export(
    model=model,
    args=(dummy_input,),
    f='model.onnx',
    input_names=list(dummy_input.keys()),
    output_names=['output'],
    dynamic_axes={
        'input_ids': {0: 'batch'},
        'attention_mask': {0: 'batch'},
        'equation_ids': {0: 'batch'},
        'decoder_input_ids': {0: 'batch'}  # Add dynamic axis for decoder_input_ids
    }
)

# Path to the ONNX model file
onnx_model_path = 'model.onnx'

# Start netron to visualize the ONNX model
netron.start(onnx_model_path)

writer.close()

#to launch tensorboard and netron from AWS, enter following in local terminal (update with your own details):
# "ssh -x -i name_of_your_aws_key.pem 6006:localhost:6006 -L 8080:localhost:8080 ubuntu@ip_address_of_your_instance"

#then in remote terminal enter:
# "tensorboard --logdir ./logs"
#then open "http://localhost:6006/" in local web browser for tensorboard and "http://localhost:8080/" for netron graph


