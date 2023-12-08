#note - Adam, 1e-5, batch size 16, 200 epochs produced 4 correct answers from first 5 records in val set
#use train.csv with 9000 records

from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from torch.utils.data import Dataset
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

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
#in the training file we would swap CustomDataset for create_dataloader

#use the dataLoader for training
model.to(device)

#define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy='epoch',
    eval_steps=500,
    num_train_epochs=200,
    predict_with_generate=True
)

writer = SummaryWriter(log_dir=training_args.logging_dir)

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

#training Loop
for epoch in range(training_args.num_train_epochs):
    #training
    trainer.train()

    #save model checkpoint
    if epoch % training_args.save_steps == 0:
        model_checkpoint_path = f'./results/checkpoint-{epoch}'
        trainer.save_model(model_checkpoint_path)

    #histograms, model weights
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

#close tensorBoard writer
writer.close()

#to launch tensorboard from AWS, enter following in local terminal (update with your own details):
# "ssh -x -i name_of_your_aws_key.pem -L 6006:localhost:6006 ubuntu@ip_address_of_your_instance"
#then in remote terminal enter:
# "tensorboard --logdir ./logs"
#then open "http://localhost:6006/" in local web browser

#display training metrics
train_metrics = trainer.evaluate()
print(f"Training metrics: {train_metrics}")

#show first five records/predictions from val dataset
for i in range(5):
    dict_item = val_dataset[i]
    input_ids = dict_item['input_ids']

    #get validation question and true answer
    val_question = val_dataset.questions[i]
    true_answer = val_dataset.answers[i]

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
word_problem = "Paul has 3.0 books. He gives 1.0 book to Amelia. How many books does Paul have now?"

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