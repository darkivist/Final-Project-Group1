import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge

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
output_dir = "/home/ubuntu/Code/flan-t5-results/checkpoint-38000/"
tokenizer = T5Tokenizer.from_pretrained(output_dir)
model = T5ForConditionalGeneration.from_pretrained(output_dir)


#################################################################################################

## Read the data

test = pd.read_csv('SVAMP_CSV_Testing.csv')
test['input'] = test['Body']  + test['Question']
test['Equation'] = test['Equation'].apply(lambda x: f'X={x.replace(" ", "")}')


print(test['Equation'])
predictions_list = []
labels_list = []
x=0
y=0
for index, row in test.iterrows():
    x = x+1
    print(x)
    input_text = row['input']

    # Preprocess input
    input_ids = preprocess_input(input_text, tokenizer)

    # Generate prediction
    prediction = generate_prediction(model, input_ids, tokenizer)

    if prediction == row['Equation']:
        y = y+1

    # Store prediction and actual label
    predictions_list.append(prediction)
    labels_list.append(row['Equation'])  # Assuming 'equation' is the column with actual labels

print(y)
# Calculate BLEU score
predictions_tokens = [prediction.split() for prediction in predictions_list]
labels_tokens = [label.split() for label in labels_list]

predictions_tokens_flat = [token for sublist in predictions_tokens for token in sublist]
labels_tokens_flat = [token for sublist in labels_tokens for token in sublist]

# Calculate BLEU score
smoothie = SmoothingFunction().method1

bleu_score = corpus_bleu([labels_tokens_flat], [predictions_tokens_flat], smoothing_function=smoothie)


print("BLEU Score:", bleu_score)


rouge = Rouge()

rouge_scores = rouge.get_scores(predictions_list, labels_list, avg=True)

print("ROUGE Scores:", rouge_scores)