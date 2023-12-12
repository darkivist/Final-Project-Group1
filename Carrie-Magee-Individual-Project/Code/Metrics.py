#%%
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel
from nltk.translate.bleu_score import corpus_bleu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MathWordProblemDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_sequence_length):
        self.mwp = dataframe["Question"].tolist()
        self.equation = dataframe["Equation"].tolist()
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.mwp)

    def __getitem__(self, idx):
        mwp_text = self.mwp[idx]
        equation_text = self.equation[idx]

        inputs = self.tokenizer(
            mwp_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            equation_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
        }

# Loading Data
df = pd.read_csv('/home/ubuntu/NLP_Main/Final-Project-Group1/Code/cleaned_dataset.csv')

checkpoint = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

max_sequence_length = 532  # Adjust this based on your data and model requirements

math_word_problem_dataset = MathWordProblemDataset(df, tokenizer, max_sequence_length)
data_loader = DataLoader(math_word_problem_dataset, batch_size=2, shuffle=True)

# Model
model = GPT2LMHeadModel.from_pretrained(checkpoint).to(device)

# Training
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 5
generated_equations = []

for epoch in range(num_epochs):
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Generate equations for evaluation
        with torch.no_grad():
            generated_ids = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2)
            generated_equations.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids])

        print(f"Epoch: {epoch + 1}, Batch Loss: {loss.item()}")

# Calculate BLEU score
reference_equations = df["Equation"].tolist()
bleu_score = corpus_bleu([[ref.split()] for ref in reference_equations], [gen.split() for gen in generated_equations])

print("BLEU Score:", bleu_score)

# %%
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel

class MathWordProblemDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_sequence_length):
        self.mwp = dataframe["Question"].tolist()
        self.equation = dataframe["Equation"].tolist()
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.mwp)

    def __getitem__(self, idx):
        mwp_text = self.mwp[idx]
        equation_text = self.equation[idx]

        inputs = self.tokenizer(
            mwp_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            equation_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
        }

# Loading Data
df = pd.read_csv('/home/ubuntu/NLP_Main/Final-Project-Group1/Code/cleaned_dataset.csv')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

max_sequence_length = 50  # Adjust this based on your data and model requirements
math_word_problem_dataset = MathWordProblemDataset(df, tokenizer, max_sequence_length)
data_loader = DataLoader(math_word_problem_dataset, batch_size=4, shuffle=True)

# Model
model = GPT2LMHeadModel.from_pretrained(checkpoint).to(device)

# Training
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training loop with gradient accumulation
accumulation_steps = 2  # Accumulate gradients over 4 batches before updating

num_epochs = 5
for epoch in range(num_epochs):
    for i, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Accumulate gradients
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            # Update parameters every accumulation_steps
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch: {epoch + 1}, Batch Loss: {loss.item()}")

# %%
