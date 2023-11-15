import torch
from transformers import EncoderDecoderModel, BertTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
import pandas as pd
import re

#load and preprocess data
data = pd.read_csv('test_data.csv')  # Replace 'your_data.csv' with your CSV file path

#function to replace 'numberX' placeholders with '[NUM]' token
def replace_number_placeholders(text):
    return re.sub(r'number\d+', '[NUM]', text)

#replace number placeholders
data['Processed_Ques'] = data['Ques'].apply(replace_number_placeholders)

questions = data['Processed_Ques'].tolist()
answers = data['Answer'].apply(lambda x: str(x)).tolist()

#initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#tokenize input sequences
tokenized_inputs = tokenizer(questions, return_tensors='pt', padding=True, truncation=True)

#tokenize output sequences
tokenized_outputs = tokenizer(answers, return_tensors='pt', padding=True, truncation=True)

#combine tokenized inputs and outputs into a list of dictionaries
dataset = [
    {
        'input_ids': tokenized_inputs['input_ids'][i],
        'attention_mask': tokenized_inputs['attention_mask'][i],
        'decoder_input_ids': tokenized_outputs['input_ids'][i],
        'decoder_attention_mask': tokenized_outputs['attention_mask'][i],
        'labels': tokenized_outputs['input_ids'][i].clone()  # labels
    }
    for i in range(len(tokenized_inputs['input_ids']))
]

#define encoder-decoder model
model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')

#training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy='steps',
    eval_steps=500,
    num_train_epochs=5,  # Increase number of epochs to 5
    predict_with_generate=True
)

#define validation dataset (using the same dataset for simplicity, adjust later)
eval_dataset = dataset

#define optimizer and learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Define Seq2SeqTrainer with evaluation dataset
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    optimizers=(optimizer, None)
)

#train model
for epoch in range(training_args.num_train_epochs):
    print(f"Epoch {epoch + 1}/{training_args.num_train_epochs}")
    trainer.train()

    #display training metrics
    train_metrics = trainer.evaluate()
    print(f"Training metrics: {train_metrics}")
