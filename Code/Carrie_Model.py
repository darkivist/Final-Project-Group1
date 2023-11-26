
# %%
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel

class MathWordProblemDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_sequence_length):
        self.mwp = dataframe["Question"].tolist()
        self.equation = dataframe["Equation"].tolist()
        self.tokenizer = tokenizer
        
        # Calculate max_sequence_length dynamically
        max_mwp_length = max(len(self.tokenizer(mwp)["input_ids"]) for mwp in self.mwp)
        max_equation_length = max(len(self.tokenizer(eq)["input_ids"]) for eq in self.equation)
        self.max_sequence_length = max(max_mwp_length, max_equation_length)

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



math_word_problem_dataset = MathWordProblemDataset(df, tokenizer, max_sequence_length)
data_loader = DataLoader(math_word_problem_dataset, batch_size=8, shuffle=True)

# Model
model = GPT2LMHeadModel.from_pretrained(checkpoint).to(device)

# Training
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

# Training loop
num_epochs = 90
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    average_loss = total_loss / len(data_loader)
    print(f"Epoch: {epoch + 1}, Average Loss: {average_loss}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        #print(f"Epoch: {epoch + 1}, Batch Loss: {loss.item()}")
#%%


import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
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
            "input_ids": inputs["input_ids"].squeeze().to(device),
            "attention_mask": inputs["attention_mask"].squeeze().to(device),
            "labels": labels["input_ids"].squeeze().to(device),
        }

# Loading Data
df = pd.read_csv('/home/ubuntu/NLP_Main/Final-Project-Group1/Code/cleaned_dataset.csv')

checkpoint = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

max_sequence_length = 532  # Adjust this based on your data and model requirements

math_word_problem_dataset = MathWordProblemDataset(df, tokenizer, max_sequence_length)
data_loader = DataLoader(math_word_problem_dataset, batch_size=8, shuffle=True)

# Model
model = GPT2LMHeadModel.from_pretrained(checkpoint).to(device)

# Training
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 6
for epoch in range(num_epochs):
    for batch in data_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch + 1}, Batch Loss: {loss.item()}")


# %%
# Input text to test
input_text = "Carrie has 6 pineapples. She ate 3. How many pineapples does she have left?"

# Tokenize the input
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=532, truncation=True).to(device)

# Generate output from the model
output_ids = model.generate(
    input_ids,
    max_length=532,
    num_beams=10,
    no_repeat_ngram_size=2,
    top_k=20,
    top_p=0.95,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id,  # Set pad token ID
    attention_mask=torch.ones(input_ids.shape, device=device),  # Set attention mask
)

# Decode the generated output
generated_equation = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print the results
print("Input Math Word Problem:")
print(input_text)
print("\nGenerated Equation:")
print(generated_equation)
del input_ids, attention_mask, labels, outputs, loss
torch.cuda.empty_cache()

# %%
### USING OPTUNA FOR HYPERPARAMETER TUNING
import optuna

def objective(trial):
    # Define hyperparameters to optimize
    lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 12])

    # Model, tokenizer, and data loader setup (keep this outside the objective function)
    model = GPT2LMHeadModel.from_pretrained(checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    math_word_problem_dataset = MathWordProblemDataset(df, tokenizer, max_sequence_length)
    data_loader = DataLoader(math_word_problem_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    num_epochs = 6
    for epoch in range(num_epochs):
        for batch in data_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Return a metric to optimize (you might want to return a validation metric)
    return loss.item()

# %%
study = optuna.create_study(direction='minimize')  # or 'maximize' depending on your goal
study.optimize(objective, n_trials=100)

best_params = study.best_params
best_loss = study.best_value

print(f"Best hyperparameters: {best_params}")
print(f"Best loss: {best_loss}")


# %%
del input_ids, attention_mask, labels, outputs, loss
torch.cuda.empty_cache()

# %%
