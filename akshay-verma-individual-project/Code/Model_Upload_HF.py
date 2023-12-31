

from transformers import T5Tokenizer, T5ForConditionalGeneration
from huggingface_hub import Repository, HfFolder, HfApi

# Set your Hugging Face API token
api_token = "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"

# Set the model path
output_dir = "/home/ubuntu/Code/flan-t5-results/checkpoint-38000/"
tokenizer = T5Tokenizer.from_pretrained(output_dir)
model = T5ForConditionalGeneration.from_pretrained(output_dir)



model.push_to_hub("averma1010/T5_Math_Equation")
tokenizer.push_to_hub("averma1010/T5_Math_Equation")
