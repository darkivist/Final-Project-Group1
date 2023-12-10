#thanks for this code, Akshay!

from transformers import T5Tokenizer, T5ForConditionalGeneration
from huggingface_hub import Repository, HfFolder, HfApi

# Set your Hugging Face API token
api_token = "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"

# Set the model path
output_dir = "/Users/paulkelly/Desktop/checkpoint-73000"
tokenizer = T5Tokenizer.from_pretrained(output_dir)
model = T5ForConditionalGeneration.from_pretrained(output_dir)

model.push_to_hub("darkivist/t5_base_math_problems")
tokenizer.push_to_hub("darkivist/t5_base_math_problems")