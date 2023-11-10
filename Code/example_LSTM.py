#%%
#INSTALLING PACKAGES
import torch
import torch.nn as nn
import torch.optim as optim 

#%%
#DEFINING LSTM MODEL
class MWPSolver(nn.Module): #our model class MWPSolver inherits from "nn.Module" from PyTorch. It provides functionailty to our model.
    def __init__(self, input_size, hidden_size, output_size): #specifying paramters 
        super(MWPSolver, self).__init__()  
        self.embedding = nn.Embedding(input_size, hidden_size) #this layer is used to convert input indices into dense vectors of fixed size (hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size) #creates LSTM layer where the input size is 'hidden_size' and the output size (hidden state size) is also 'hidden_size"
        #the first hidden_size is usually the size of the input features and the hidden state
        #the second hidden_size specifies the size of the hidden state the output of the LSTM layer 
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2) #often used as the final layer in classification problems 

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        lstm_out, _ = self.lstm(embedded) #concise way of only extracting the output tensor ('lstm_out')
        output = self.fc(lstm_out)
        output = self.softmax(output)
        return output

#%%
#EXAMPLE PROBLEM
word_problem = "John has 5 apples, and he buys 3 more. How many apples does John have now?"
vocab = {'John': 0, 'has': 1, '5': 2, 'apples': 3, 'and': 4, 'he': 5, 'buys': 6, '3': 7, 'more': 8, 'How': 9, 'many': 10, 'does': 11, 'now': 12, '?': 13}
#getting indices
word_indices = [vocab[word.lower()] for word in word_problem.split()]

#Convert to PyTorch tensor
input_sequence = torch.tensor(word_indices).view(-1, 1) #reshape to have single column; reshaping is useful when we want to treat each element in 'word_indices' as a 
#separate instance or time step in a sequence 

#Initialize model parameters
input_size = len(vocab)
hidden_size = 64
output_size = len(vocab)
model = MWPSolver(input_size, hidden_size, output_size)

#%%
#Defining loss and optimizer
# Define loss and optimizer
criterion = nn.NLLLoss() #Negative Log Likelihood Loss is commonly used with models that have a softmax activation function in the output layer
optimizer = optim.SGD(model.parameters(), lr=0.01)

#%% 
#TRAINING THE DATA 
for epoch in range(100): #change based of data 
    optimizer.zero_grad() #used to zero out the gradients of the model parameters 
    output = model(input_sequence)

     # Assuming target equation is provided (ground truth)
    target_equation = "5 + 3"
    target_indices = [vocab[word.lower()] for word in target_equation.split()]
    target_tensor = torch.tensor(target_indices).view(-1)

    loss = criterion(output.view(-1, len(vocab)), target_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

#%%
#TESTING MODEL ON NEW DATA
test_word_problem = "Mary has 8 books. She gives 3 books to her friend. How many books does Mary have now?"

test_word_indices = [vocab[word.lower()] for word in test_word_problem.split()]
test_input_sequence = torch.tensor(test_word_indices).view(-1, 1)

with torch.no_grad():
    test_output = model(test_input_sequence)
    _, predicted_indices = torch.max(test_output, 2)

    # Convert predicted indices back to words
    predicted_equation = ' '.join([word for idx in predicted_indices[0] for word, idx_map in vocab.items() if idx == idx_map])
    print(f'Test Word Problem: {test_word_problem}')
    print(f'Predicted Equation: {predicted_equation}')