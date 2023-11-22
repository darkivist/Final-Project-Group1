from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, source_texts, target_texts, tokenizer, max_source_length=512, max_target_length=512):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = str(self.source_texts[idx])
        target_text = str(self.target_texts[idx])

        encoding = self.tokenizer(
            text=source_text,
            text_pair=target_text,
            padding='max_length',
            max_length=self.max_source_length,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            text=target_text,
            padding='max_length',
            max_length=self.max_target_length,
            return_tensors="pt"
        )["input_ids"]

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": labels.flatten(),
        }

# Load your dataset
# Assuming you have a DataFrame named 'data' containing columns: 'Question', 'Equation', 'Answer'
# Adjust this according to your actual data loading process
data = pd.read_csv('train.csv')

#function to replace number0 (etc) values with actual numbers
def replace_number_placeholders(row):
    def replace_number(match):
        number_index = int(match.group(1))  #extract number index
        return str(row['Numbers'].split(' ')[number_index])

    return re.sub(r'number(\d+)', replace_number, row['Question'])

#ensure 'ques' is a string
data['Question'] = data['Question'].astype(str)

#replace number placeholders
data['Processed_Question'] = data.apply(replace_number_placeholders, axis=1)

# Prepare the dataset for translation (Question -> Equation, Answer as target)
source_texts = data['Processed_Question'].tolist()
target_texts = data['Equation'].tolist()
answers = data['Answer'].tolist()

# Split the dataset into train and validation sets
train_source_texts, val_source_texts, train_target_texts, val_target_texts = train_test_split(
    source_texts, target_texts, test_size=0.1, random_state=42
)

# Initialize the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# Create the CustomDataset instances
train_dataset = CustomDataset(train_source_texts, train_target_texts, tokenizer)
val_dataset = CustomDataset(val_source_texts, val_target_texts, tokenizer)

# Define the training arguments
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
    predict_with_generate=True)

# Define the Seq2Seq trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
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
#print("tokenized input:",tokenized_input)

#generate answer
output = model.generate(input_ids=tokenized_input['input_ids'],
                        attention_mask=tokenized_input['attention_mask'],
                        max_length=200)

#decode the generated output tokens to text
prediction = tokenizer.decode(output[0], skip_special_tokens=True)

print("Answer:", prediction)
