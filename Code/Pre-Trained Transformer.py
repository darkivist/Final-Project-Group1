'''
Just using a pre-trained model here to solve the math word problems. This is definitely not what we want to present.
I think this might give us a good benchmark score, but in all likelihood we won't get anywhere the same accuracy.
'''
import pandas as pd
from transformers import pipeline
from nltk.translate.bleu_score import corpus_bleu


df = pd.read_csv('SVAMP_CSV.csv')
df['problem'] = df['Body'] + df['Question']


problems = df['problem'].tolist()
true_answers = df['Equation'].tolist()


## GPT-Neo 1.3B is a transformer model designed using EleutherAI's replication of the GPT-3 architecture.
model = pipeline('text2text-generation', model='EleutherAI/gpt-neo-1.3B')


def solve_math_word_problem(word_problem):
    equation = model(word_problem, max_length=30, num_return_sequences=1)[0]['generated_text'].strip()
    return equation


predicted_answers = [solve_math_word_problem(problem) for problem in problems]

correct_predictions = [1 if pred == true else 0 for pred, true in zip(predicted_answers, true_answers)]
accuracy = sum(correct_predictions) / len(correct_predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

true_answers_tokens = [[word.lower() for word in answer.split()] for answer in true_answers]
predicted_answers_tokens = [[word.lower() for word in answer.split()] for answer in predicted_answers]

# Calculate BLEU score
bleu_score = corpus_bleu([true_answers_tokens], [predicted_answers_tokens])
print(f"BLEU Score: {bleu_score * 100:.2f}%")

