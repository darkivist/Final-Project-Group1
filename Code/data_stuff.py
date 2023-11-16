import pandas as pd

# Read the dataset
dataset_path = '/home/ubuntu/NLP_Main/Final-Project-Group1/Code/SVAMP_CSV.csv'
df = pd.read_csv(dataset_path)

# Join 'Body' and 'Question' columns
df['Question'] = df['Body'] + ' ' + df['Question']

# Drop 'Body', 'Question', 'Answer', and 'Type' columns
df_cleaned = df.drop(["ID",'Body', 'Answer', 'Type'], axis=1)

# Save the cleaned dataset to a new CSV file
df_cleaned.to_csv('cleaned_dataset.csv', index=False)
