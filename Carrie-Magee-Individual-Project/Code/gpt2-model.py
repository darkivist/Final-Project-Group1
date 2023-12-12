
# %%
### DO NOT LOOK AT THIS ONE THIS WAS A TEST
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
num_epochs = 5
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
### USING OPTUNA FOR HYPERPARAMETER TUNING
import optuna

def objective(trial):
    # Define hyperparameters to optimize
    lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])

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
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_loss = study.best_value

print(f"Best hyperparameters: {best_params}")
print(f"Best loss: {best_loss}")


# %%
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, AutoTokenizer
from sklearn.model_selection import train_test_split

# Assuming you have already defined the MathWordProblemDataset and other necessary components

# Load the new dataset
new_df = pd.read_csv("/home/ubuntu/NLP_Main/Final-Project-Group1/Code/cleaned_dataset.csv")

# Keep only the "Question" and "Equation" columns

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(new_df, test_size=0.2, random_state=42)

# Set the best hyperparameters
best_lr = study.best_params['lr']
best_batch_size = study.best_params['batch_size']

# Model, tokenizer, and data loader setup
model = GPT2LMHeadModel.from_pretrained(checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

math_word_problem_dataset_train = MathWordProblemDataset(train_df, tokenizer, max_sequence_length)
math_word_problem_dataset_test = MathWordProblemDataset(test_df, tokenizer, max_sequence_length)

data_loader_train = DataLoader(math_word_problem_dataset_train, batch_size=best_batch_size, shuffle=True)
data_loader_test = DataLoader(math_word_problem_dataset_test, batch_size=best_batch_size, shuffle=False)

# Optimizer setup with the best learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=best_lr)

# Training loop
num_epochs = 6
for epoch in range(num_epochs):
    model.train()
    for batch in data_loader_train:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save the trained model if needed
torch.save(model.state_dict(), 'best_model.pth')

# Now you can use the trained model to generate predictions on the test set
model.eval()
all_predictions = []

with torch.no_grad():
    for test_batch in data_loader_test:
        test_input_ids = test_batch["input_ids"]
        test_attention_mask = test_batch["attention_mask"]
        predictions = model.generate(test_input_ids, attention_mask=test_attention_mask, max_length=532)
        decoded_predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
        all_predictions.extend(decoded_predictions)

# Perform evaluation based on your specific task and metrics
# ...

# Print or visualize the results
print("Predictions:", all_predictions)

# %%
