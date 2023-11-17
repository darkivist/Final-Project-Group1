import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.nn.utils.rnn import pad_sequence


def tokenize(text):
    tokens = text.split()
    return tokens

class MathSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(MathSeq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        print("Input Tensor:", x)
        embedded = self.embedding(x)
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(embedded)
        decoder_output, _ = self.decoder(embedded, (encoder_hidden, encoder_cell))
        output = self.fc(decoder_output)
        return output

class MathProblemDataset(Dataset):
    def __init__(self, problems, equations, tokenizer):
        self.problems = problems
        self.equations = equations
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        problem = self.tokenizer(self.problems.iloc[idx])
        equation = self.tokenizer(self.equations.iloc[idx])
        return {'problem': problem, 'equation': equation}

def collate_fn(batch):
    problems = []
    equations = []

    for item in batch:
        try:
            problem = [int(idx) for idx in item['problem'] if idx.isdigit()]
            equation = [int(idx) for idx in item['equation'] if idx.isdigit()]

            problems.append(torch.LongTensor(problem))
            equations.append(torch.LongTensor(equation))
        except ValueError as e:
            print(f"Error processing batch: {e}")
            print("Batch item:", item)

    # Pad sequences to the length of the longest sequence in the batch
    problems_padded = pad_sequence(problems, batch_first=True, padding_value=0)
    equations_padded = pad_sequence(equations, batch_first=True, padding_value=0)

    # Flatten the target tensor
    equations_padded_flat = equations_padded.view(-1)

    return {'problem': problems_padded, 'equation': equations_padded_flat}

# Load data
df = pd.read_csv('SVAMP_CSV.csv')
df['problems'] = df['Body'] + df['Question']

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Instantiate dataset and dataloader
math_dataset = MathProblemDataset(train_data['problems'], train_data['Equation'], tokenize)
batch_size = 64
train_loader = DataLoader(math_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Initialize the model
vocab_size = 10000  # Adjust based on your data
embedding_dim = 256
hidden_dim = 512
num_layers = 3

model = MathSeq2Seq(vocab_size, embedding_dim, hidden_dim, num_layers)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch in train_loader:
        input_data = batch['problem']
        target = batch['equation']

        # Forward pass
        output = model(input_data)

        # Reshape the output tensor to match the target tensor
        output = output.view(-1, vocab_size)

        # Flatten the target tensor
        target_flat = target.view(-1)

        # Compute the loss
        loss = criterion(output, target_flat)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()