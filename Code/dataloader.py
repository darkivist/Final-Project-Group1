from torch.utils.data import DataLoader
from dataclass import CustomDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer
import re

def replace_number_placeholders(row):
    def replace_number(match):
        number_index = int(match.group(1))  #extract number index
        return str(row['Numbers'].split(' ')[number_index])

    return re.sub(r'number(\d+)', replace_number, row['Question'])

def load_data(file_path):
    data = pd.read_csv(file_path)
    
    #ensure 'ques' is a string
    data['Question'] = data['Question'].astype(str)
    #replace number placeholders
    data['Processed_Question'] = data.apply(replace_number_placeholders, axis=1)

    # train/val split
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # select fields needed for train and val
    train_questions = train_data['Processed_Question'].tolist()
    train_equations = train_data['Equation'].tolist()
    train_answers = train_data['Answer'].tolist()

    val_questions = val_data['Processed_Question'].tolist()
    val_equations = val_data['Equation'].tolist()
    val_answers = val_data['Answer'].tolist()

    return (train_questions, train_equations, train_answers), (val_questions, val_equations, val_answers)

def create_dataloader(questions, equations, answers, tokenizer, batch_size=16, max_length=128):
    dataset = CustomDataset(questions, equations, answers, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
