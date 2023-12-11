import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pickle

def preprocess_input(input_text, tokenizer, max_length=128):
    input_ids = tokenizer.encode_plus(
        "translate Question to Equation: " + input_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return input_ids

def generate_prediction(model, input_ids, tokenizer, max_length=128):
    output_ids = model.generate(
        input_ids['input_ids'],
        attention_mask=input_ids['attention_mask'],
        max_length=max_length,
        num_beams=4,  # You can adjust the beam size for beam search
        length_penalty=2.0,  # You can adjust the length penalty
        early_stopping=True,
        no_repeat_ngram_size=2, #You can adjust the no_repeat_ngram_size
        decoder_start_token_id=tokenizer.pad_token_id #
    )
    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return prediction

# Load pretrained model and tokenizer

output_dir = "/home/ubuntu/Code/flan-t5-results/checkpoint-38000/"
tokenizer = T5Tokenizer.from_pretrained(output_dir)
model = T5ForConditionalGeneration.from_pretrained(output_dir)

# Example input for inference
input_text = "The perimeter of a rectangular garden is 28 meters, and its length is 8 meters. What is the width of the garden?"

# Preprocess input
input_ids = preprocess_input(input_text, tokenizer)

# Generate prediction
prediction = generate_prediction(model, input_ids, tokenizer)

# Display the result
print("Input Text:", input_text)
print("Predicted Equation:", prediction)


