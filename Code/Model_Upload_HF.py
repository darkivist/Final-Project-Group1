import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import T5Tokenizer, T5ForConditionalGeneration
from huggingface_hub import notebook_login


output_dir = "./saved_model_T5_Equation"
tokenizer = T5Tokenizer.from_pretrained(output_dir)
model = T5ForConditionalGeneration.from_pretrained(output_dir)



notebook_login()



model.push_to_hub("averma1010/T5_Math_Equation")