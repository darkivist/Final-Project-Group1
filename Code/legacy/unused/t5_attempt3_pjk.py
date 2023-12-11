from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#define custom dataset that includes questions, equations, and answers
class CustomDataset(Dataset):
    def __init__(self, questions, equations, answers, tokenizer, max_length=128):
        self.questions = questions
        self.equations = equations
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = str(self.questions[idx])
        equation = str(self.equations[idx])
        answer = str(self.answers[idx])

        #tokenize inputs and outputs
        inputs = self.tokenizer.encode_plus(
            "translate Question to Equation: " + question,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        equation_encoding = self.tokenizer.encode(
            equation,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = self.tokenizer.encode(
            answer,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs.input_ids.flatten(),
            "attention_mask": inputs.attention_mask.flatten(),
            "equation_ids": equation_encoding.flatten(),
            "labels": labels.flatten()
        }

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

#train/val split
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

#select fields we need for train and val
train_questions = train_data['Processed_Question'].tolist()
train_equations = train_data['Equation'].tolist()
train_answers = train_data['Answer'].tolist()

val_questions = val_data['Processed_Question'].tolist()
val_equations = val_data['Equation'].tolist()
val_answers = val_data['Answer'].tolist()

#initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

#create custom train/val datasets
train_dataset = CustomDataset(train_questions, train_equations, train_answers, tokenizer)
val_dataset = CustomDataset(val_questions, val_equations, val_answers, tokenizer)

#use dataloader to handle batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

#use the dataLoader for training
model.to(device)

#set optimizer and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#set number of epochs
num_epochs = 1

#training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        equation_ids = batch['equation_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=equation_ids  # use the equation as the initial input for the decoder
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        #validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(input_ids=batch['input_ids'].to(device),
                                attention_mask=batch['attention_mask'].to(device),
                                labels=batch['labels'].to(device))

                loss = outputs.loss
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

print('Training complete!')

#function to tokenize sample problem for testing
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
output = model.generate(input_ids=tokenized_input['input_ids'].to(device),
                        attention_mask=tokenized_input['attention_mask'].to(device),
                        max_length=200)

#decode the generated output tokens to text
prediction = tokenizer.decode(output[0], skip_special_tokens=True)

print("Answer:", prediction)