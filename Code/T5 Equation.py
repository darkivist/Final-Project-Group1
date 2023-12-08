import pickle
import torch
torch.cuda.empty_cache()
from nltk.translate.bleu_score import corpus_bleu
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
import json

import torch
from torch.utils.data import Dataset
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os
os.environ["WANDB_DISABLED"] = "true"

data = pd.read_pickle('MAWPS_Augmented.pkl')
print(data)


def compute_metrics(p, tokenizer, dataset):
  predictions = p.predictions
  decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
  references = [str(answer) for answer in dataset.answers]

  # Calculate and return the metric of interest, e.g., BLEU score
  bleu_score = corpus_bleu(decoded_preds, [references]).score
  return {"bleu": bleu_score}

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

    # tokenize inputs and outputs
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





# function to replace number0 (etc) values with actual numbers




# ensure 'ques' is a string
data['Question'] = data['Question'].astype(str)


# train/val split
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# select fields we need for train and val
train_questions = train_data['Question'].tolist()
train_equations = train_data['Equation'].tolist()
train_answers = train_data['Equation'].tolist()

val_questions = val_data['Question'].tolist()
val_equations = val_data['Equation'].tolist()
val_answers = val_data['Equation'].tolist()

# initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

# create custom train/val datasets
train_dataset = CustomDataset(train_questions, train_equations, train_answers, tokenizer)
val_dataset = CustomDataset(val_questions, val_equations, val_answers, tokenizer)
# in the training file we would swap CustomDataset for create_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# use the dataLoader for training
model.to(device)

# define training arguments
training_args = Seq2SeqTrainingArguments(
  output_dir='./flan-t5-results',
  per_device_train_batch_size=16,
  per_device_eval_batch_size=16,
  logging_dir='./logs',
  logging_steps=100,
  save_steps=1000,
  evaluation_strategy='epoch',
  eval_steps=500,
  num_train_epochs=20,
  predict_with_generate=True,
)



# define optimizer and instantiate Seq2SeqTrainer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
trainer = Seq2SeqTrainer(
  model=model,
  args=training_args,
  train_dataset=train_dataset,
  eval_dataset=val_dataset,
  tokenizer=tokenizer,
  optimizers=(optimizer, None),
)

# train model
trainer.train(resume_from_checkpoint=True)


# Save the trained model
output_dir = "./saved_model_flan-t5-base_Equation"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Save additional configuration information
config = {
    "max_length": 128,  # You may adjust this based on your needs
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    # Add any other configuration parameters you want to save
}

with open(os.path.join(output_dir, "config.json"), "w") as config_file:
    json.dump(config, config_file)



# display training metrics
train_metrics = trainer.evaluate()
print(f"Training metrics: {train_metrics}")

# show first five records/predictions from val dataset
for i in range(5):
  dict_item = val_dataset[i]
  input_ids = dict_item['input_ids']

  # get validation question and true answer
  val_question = val_dataset.questions[i]
  true_answer = val_dataset.answers[i]

  # reshape input_ids to add batch and sequence length dimensions
  input_ids = input_ids.unsqueeze(0).to(device)

  # generate prediction
  outputs = model.generate(input_ids)
  prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

  print(f"Validation Question: {val_question}")
  print(f"Predicted Text: {prediction}")
  print(f"True Answer: {true_answer}")





# function to tokenize sample problem for testing
def preprocess_word_problem(problem_text):
  # tokenize the problem text
  tokenized_problem = tokenizer(problem_text, return_tensors='pt', padding=True, truncation=True)
  return tokenized_problem


# sample word problem
word_problem = "Samantha has 24 apples. She wants to share them equally among 6 friends. How many apples will each friend get?"

print("Word problem:", word_problem)

# preprocess word problem
device = 'cuda:0'
tokenized_input = preprocess_word_problem(word_problem)
tokenized_input = {key: tensor.to(device) for key, tensor in tokenized_input.items()}
# print("tokenized input:",tokenized_input)


# generate answer
output = model.generate(input_ids=tokenized_input['input_ids'].to(device),
                        attention_mask=tokenized_input['attention_mask'].to(device),
                        max_length=200)

# decode the generated output tokens to text
prediction = tokenizer.decode(output[0], skip_special_tokens=True)

print("Answer:", prediction)