#%%
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
import pandas as pd
from dataclass import CustomDataset  
from dataloader import load_data, create_dataloader
from nltk.translate.bleu_score import corpus_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_data = pd.read_csv('test.csv') #havent made test.csv yet

# Preprocess test data similar to training/validation data
test_questions = test_data['Processed_Question'].tolist()
test_equations = test_data['Equation'].tolist()
test_answers = test_data['Answer'].tolist()

checkpoint_path = '/path/to/checkpoint-97000'

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("path_to_pretrained_model").to(device)

test_dataset = CustomDataset(test_questions, test_equations, test_answers, tokenizer)


test_dataloader = create_dataloader(test_dataset, batch_size=16)  


trainer = Seq2SeqTrainer(
    model=model,
    args=None,
    data_collator=None,
    tokenizer=tokenizer,
    compute_metrics=None  
)

# Evaluate metrics on the test set
test_metrics = trainer.predict(test_dataloader)

# Display test metrics
print(f"Test metrics: {test_metrics}")

#BLEU Score
test_predictions = trainer.predict(test_dataloader).predictions
decoded_predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in test_predictions]

# Extract reference sentences (ground truth) from your test dataset
references = [[answer] for answer in test_answers]

# Calculate BLEU score
bleu_score = corpus_bleu(references, decoded_predictions)
print(f"BLEU Score: {bleu_score * 100:.2f}")
