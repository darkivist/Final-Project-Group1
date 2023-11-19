#%%    
#### OTHER MODEL
import pandas as pd
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.nn import CrossEntropyLoss

# Load the data
df = pd.read_csv('/home/ubuntu/NLP_Main/Final-Project-Group1/Code/cleaned_dataset.csv')

# Extract input and target expressions
input_exps = list(df['Question'].values)
target_exps = list(df['Equation'].values)


# Preprocess input and target expressions
def preprocess_input(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,’])", r" \1 ", sentence)
    sentence = re.sub(r"([0-9])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = sentence.rstrip().strip()
    return sentence

def preprocess_target(sentence):
    sentence = sentence.lower().strip()
    return sentence

preprocessed_input_exps = list(map(preprocess_input, input_exps))
preprocessed_target_exps = list(map(preprocess_target, target_exps))


# Tokenize text
tokenizer = T5Tokenizer.from_pretrained("t5-small")
# Reduce batch size
batch_size = 4  # Adjust according to your system's capacity
tokenized_inputs = tokenizer(preprocessed_input_exps, return_tensors="pt", padding=True, truncation=True, max_length=64, max_batch_size=batch_size)
tokenized_targets_no_variable = tokenizer(preprocessed_target_exps, return_tensors="pt", padding=True, truncation=True, max_length=64, max_batch_size=batch_size)

#tokenized_inputs = tokenizer(preprocessed_input_exps, return_tensors="pt", padding=True, truncation=True, max_length=64)
#tokenized_targets_no_variable = tokenizer(preprocessed_target_exps, return_tensors="pt", padding=True, truncation=True, max_length=64)

# Create Seq2Seq model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss()

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    attention_mask = tokenized_inputs["attention_mask"]
    outputs = model(input_ids=tokenized_inputs["input_ids"], attention_mask=attention_mask, labels=tokenized_targets_no_variable["input_ids"])

    loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), tokenized_targets_no_variable["input_ids"].view(-1))

    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

def generate_prediction(model, tokenizer, input_text):
    model.eval()  # Set the model to evaluation mode
    input_text = preprocess_input(input_text)
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=64)["input_ids"]
    
    # Generate output from the model
    output_ids = model.generate(input_ids)
    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return decoded_output
# Replace "Your test sentence goes here." with your actual test sentence
test_sentence = "David has 48 marbles. He puts them into 4 bags. How many marbles are there in each bag?"

# Generate prediction
with torch.no_grad():  # No need for gradient computation during inference
    prediction = generate_prediction(model, tokenizer, test_sentence)

print(f"Test Sentence: {test_sentence}")
print(f"Generated Prediction: {prediction}")
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration

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
            f"translate English to Math: {mwp_text}",
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

checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

max_sequence_length = 532  # Adjust this based on your data and model requirements

math_word_problem_dataset = MathWordProblemDataset(df, tokenizer, max_sequence_length)
data_loader = DataLoader(math_word_problem_dataset, batch_size=2, shuffle=True)

# Model
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

# Training
loss_function = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_epochs = 6
accumulation_steps = 4  # Accumulate gradients over 4 steps
for epoch in range(num_epochs):
    total_loss = 0
    for i, batch in enumerate(data_loader):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Accumulate gradients
        loss = loss / accumulation_steps
        loss.backward()

        # Update parameters every accumulation_steps
        if (i + 1) % accumulation_steps == 0 or i == len(data_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    print(f"Epoch: {epoch + 1}, Average Loss: {total_loss / len(data_loader)}")


#%%
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
