import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
import seaborn as sns
import matplotlib.pyplot as plt
from sympy import sympify



def preprocess_input(input_text, tokenizer, max_length=128):
    input_ids = tokenizer.encode_plus(
        "translate Question to Equation: " + input_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return input_ids

def generate_prediction(model, input_ids, tokenizer, max_length=128):
    output_ids = model.generate(
        input_ids['input_ids'],
        attention_mask=input_ids['attention_mask'],
        max_length=max_length,
        num_beams=4,  # You can adjust the beam size for beam search
        length_penalty=2.0,  # You can adjust the length penalty
        early_stopping=True,
        no_repeat_ngram_size=2, #You can adjust the no_repeat_ngram_size
        decoder_start_token_id=tokenizer.pad_token_id #
    )
    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return prediction


# Load pretrained model and tokenizer
output_dir = "darkivist/t5_base_math_problems"
tokenizer = T5Tokenizer.from_pretrained(output_dir)
model = T5ForConditionalGeneration.from_pretrained(output_dir)


#################################################################################################

## Read the data

test = pd.read_csv('SVAMP_CSV_Testing.csv')
test['input'] = test['Body']  + test['Question']
test['Equation'] = test['Equation'].apply(lambda x: f'X={x.replace(" ", "")}')

print(test['Type'].value_counts())
print(test['Equation'])
predictions_list = []
labels_list = []
x=0
y=0
correct_predictions_by_type = {}

for index, row in test.iterrows():
    x = x + 1
    print(x)
    input_text = row['input']

    # Preprocess input
    input_ids = preprocess_input(input_text, tokenizer)

    prediction = generate_prediction(model, input_ids, tokenizer)

    # Generate prediction
    prediction = prediction.replace('(', '').replace(')', '').replace('X', '').replace("=", '').replace('x','')
    #actual_label = row['Equation'].replace('(', '').replace(')', '').replace('X=', '').replace("=", '').replace('x','')
    actual_label = row['Answer']

    try:
        # Use sympify to check if the cleaned prediction and actual label are equal
        cleaned_prediction = sympify(prediction)
        cleaned_actual_label = sympify(actual_label)

        if cleaned_prediction == cleaned_actual_label:
            y = y + 1

            prediction_type = row['Type']
            correct_predictions_by_type[prediction_type] = correct_predictions_by_type.get(prediction_type, 0) + 1

        # Store cleaned prediction and actual label
        predictions_list.append(prediction)
        labels_list.append(actual_label)  # Assuming 'equation' is the column with actual labels

    except Exception as e:
        # Handle the exception (you can print or log the error if needed)
        print(f"Error processing row {index}: {e}")
        print(row['Type'])

print(y)

for prediction_type, count in correct_predictions_by_type.items():
    print(f"{prediction_type}: {count}")



#Calculate ROUGE score

rouge = Rouge()

rouge_scores = rouge.get_scores(predictions_list, labels_list, avg=True)


print("ROUGE Scores:", rouge_scores)
rouge_scores_data = {
    'ROUGE-1': [rouge_scores['rouge-1']['f']],
    'ROUGE-2': [rouge_scores['rouge-2']['f']],
    'ROUGE-L': [rouge_scores['rouge-l']['f']]
}

# Creating a DataFrame for easier visualization
rouge_df = pd.DataFrame(rouge_scores_data)

# Plotting the ROUGE scores using a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=rouge_df, palette='viridis')
plt.title('ROUGE Scores')
plt.ylabel('Score')
plt.ylim(0, 1)  # Adjust the y-axis limit if needed
plt.show()