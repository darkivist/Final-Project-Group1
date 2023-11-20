import pandas as pd
import re
from sklearn.model_selection import train_test_split
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

#load and preprocess data
data = pd.read_csv('train.csv')

#function to replace 'numberx' placeholders with '[NUM]' token
def replace_number_placeholders(text):
    return re.sub(r'number\d+', '[NUM]', text)

#ensure 'ques' is a string
data['Ques'] = data['Ques'].astype(str)

#replace number placeholders
data['Processed_Ques'] = data['Ques'].apply(replace_number_placeholders)

#train/val split
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

train_questions = train_data['Processed_Ques'].tolist()
train_answers = train_data['Answer'].apply(lambda x: str(x)).tolist()

val_questions = val_data['Processed_Ques'].tolist()
val_answers = val_data['Answer'].apply(lambda x: str(x)).tolist()

#initializing tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

#tokenize input and output sequences
tokenized_train_inputs = tokenizer(train_questions, return_tensors='pt', padding=True, truncation=True)
tokenized_train_outputs = tokenizer(train_answers, return_tensors='pt', padding=True, truncation=True)

tokenized_val_inputs = tokenizer(val_questions, return_tensors='pt', padding=True, truncation=True)
tokenized_val_outputs = tokenizer(val_answers, return_tensors='pt', padding=True, truncation=True)

train_dataset = [
    {
        'input_ids': tokenized_train_inputs['input_ids'][i],
        'attention_mask': tokenized_train_inputs['attention_mask'][i],
        'decoder_input_ids': tokenized_train_outputs['input_ids'][i],
        'decoder_attention_mask': tokenized_train_outputs['attention_mask'][i],
        'labels': tokenized_train_outputs['input_ids'][i].clone()  # Labels
    }
    for i in range(len(tokenized_train_inputs['input_ids']))
]

val_dataset = [
    {
        'input_ids': tokenized_val_inputs['input_ids'][i],
        'attention_mask': tokenized_val_inputs['attention_mask'][i],
        'decoder_input_ids': tokenized_val_outputs['input_ids'][i],
        'decoder_attention_mask': tokenized_val_outputs['attention_mask'][i],
        'labels': tokenized_val_outputs['input_ids'][i].clone()  # Labels
    }
    for i in range(len(tokenized_val_inputs['input_ids']))
]

#define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy='steps',
    eval_steps=500,
    num_train_epochs=5,
    predict_with_generate=True
)

#define encoder-decoder model
model = T5ForConditionalGeneration.from_pretrained('t5-small')

#define optimizer and instantiate Seq2SeqTrainer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    optimizers=(optimizer, None)
)

#train model
trainer.train()

#display training metrics
train_metrics = trainer.evaluate()
print(f"Training metrics: {train_metrics}")

def preprocess_word_problem(problem_text):
    #tokenize the problem text
    tokenized_problem = tokenizer(problem_text, return_tensors='pt', padding=True, truncation=True)
    return tokenized_problem

#sample word problem
word_problem = "Paul has 3 books. He gives 1 book to Amelia. How many books does Paul have now?"

print("Word problem:", word_problem)

#preprocess word problem
device = 'cuda:0'
tokenized_input = preprocess_word_problem(word_problem)
tokenized_input = {key: tensor.to(device) for key, tensor in tokenized_input.items()}
print("tokenized input:",tokenized_input)

#wrap in batch
#batched_input = {
#    'input_ids': tokenized_input['input_ids'].unsqueeze(0),
#    'attention_mask': tokenized_input['attention_mask'].unsqueeze(0)
#}

#generate answer
output = model.generate(input_ids=tokenized_input['input_ids'],
                        attention_mask=tokenized_input['attention_mask'],
                        max_length=200)

print("output:", output)

#decode the generated output tokens to text
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print("decoded output:", decoded_output)