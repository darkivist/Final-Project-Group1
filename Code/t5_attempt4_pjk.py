#note - 1e-5 and batch size 16 gave 2 correct answers from first 5 records in val set
#continue to refine

from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import optuna

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

#define function for hyperparameter tuning
def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    num_training_epochs = trial.suggest_int('num_epochs', 1, 100)
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop', 'AdamW', 'Adadelta'])
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

    #define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir='./logs',
        logging_steps=100,
        save_steps=1000,
        evaluation_strategy='epoch',
        eval_steps=500,
        num_train_epochs=num_training_epochs,
        predict_with_generate=True
    )

    #create custom train/val datasets
    train_dataset = CustomDataset(train_questions, train_equations, train_answers, tokenizer)
    val_dataset = CustomDataset(val_questions, val_equations, val_answers, tokenizer)

    #instantiate Seq2SeqTrainer
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

    #get validation loss
    val_metrics = trainer.evaluate()
    val_loss = val_metrics['eval_loss']

    return val_loss

#perform hyperparameter studies
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

#return the best hyperparameters
best_learning_rate = study.best_params['learning_rate']
best_batch_size = study.best_params['batch_size']
num_training_epochs = study.best_params['num_epochs']
best_optimizer = study.best_params['optimizer']
print("Best learning rate:", best_learning_rate)
print("Best batch size:", best_batch_size)
print("Best number of epochs:", num_training_epochs)
print("Best optimizer:", best_optimizer)

#create custom train/val datasets
train_dataset = CustomDataset(train_questions, train_equations, train_answers, tokenizer)
val_dataset = CustomDataset(val_questions, val_equations, val_answers, tokenizer)

#use dataloader to handle batching and shuffling (not required - comment out)
#train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

#use the dataLoader for training
model.to(device)

#set optimized optimizer (lol) and optimized learning rate
optimizer = best_optimizer

#define optimized training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=best_batch_size,
    per_device_eval_batch_size=best_batch_size,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy='epoch',
    eval_steps=500,
    num_train_epochs=num_training_epochs,
    predict_with_generate=True
)
#instantiate Seq2SeqTrainer again
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    optimizers=(optimizer, None)
)

#retrain model
trainer.train()

#display training metrics
train_metrics = trainer.evaluate()
print(f"Training metrics: {train_metrics}")

#show first five records/predictions from val dataset
for i in range(5):
    dict_item = val_dataset[i]
    input_ids = dict_item['input_ids']

    #get validation question and true answer
    val_question = val_dataset.questions[i]  # Accessing the validation question directly from the dataset
    true_answer = val_dataset.answers[i]  # Accessing the true answer directly from the dataset

    #reshape input_ids to add batch and sequence length dimensions
    input_ids = input_ids.unsqueeze(0).to(device)

    #generate prediction
    outputs = model.generate(input_ids)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Validation Question: {val_question}")
    print(f"Predicted Text: {prediction}")
    print(f"True Answer: {true_answer}")


#function to tokenize sample problem for testing
def preprocess_word_problem(problem_text):
    #tokenize the problem text
    tokenized_problem = tokenizer(problem_text, return_tensors='pt', padding=True, truncation=True)
    return tokenized_problem

#sample word problem
word_problem = "Paul has 3.0 books. He gives 1.0 book to Amelia and another to Megan. How many books does Paul have now?"

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