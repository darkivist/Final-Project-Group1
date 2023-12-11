from torch.utils.data import Dataset
from transformers import T5Tokenizer

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