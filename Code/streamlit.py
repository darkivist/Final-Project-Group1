# Imports
import streamlit as st

import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
st.set_page_config(layout="wide")
from PIL import Image
import pandas as pd


# ------------Loading Model -------------------
output_dir = "averma1010/T5_Math_Equation"
tokenizer_dir = "averma1010/T5_Math_Equation"

tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir)

def load_model():
    model = T5ForConditionalGeneration.from_pretrained(output_dir)
    return model

model = load_model()


# ---------- Titles and headers -----------
st.title("NLP Group 1: Translating Math Problems")
st.subheader("Paul Kelly, Carrie Magee, Jack McMorrow, Akshay Verma")


tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introduction & Dataset", "NLP Model", "Experimental Design and Hyperparameters"  , "Results & Limitaion", "Model Demo"])



with tab1:

    
    st.subheader("Introduction")
    

    st.markdown("Language complexity gets tricky, especially when mixing math into word problems. Our group took on this challenge using transformers to simplify the complexities in math and languages")
    st.markdown("Math word problems, those short stories with real-world scenarios, need more than just basic math skills. You've got to understand the context, sentence structure, and word connections. Solving them means figuring out the problem, picking out the important info, and turning it into solvable math. Computers find this tricky due to ambiguity and understanding context.")

    st.image("https://raw.githubusercontent.com/darkivist/Final-Project-Group1/ab1a9fe0168aa35d5d37017c58f0317fac0322b9/Code/Images/stock-image.jpeg")
    st.divider()

    st.subheader("Goals of the project")
    
    st.markdown(" In our attempt to solve this problem, the multimodal capacities of transformers emerged as valuable assets. The goal of our project is to solve linguistic challenges with computational solutions, more specifically use the power of deep learning to convert word problems into solvable mathematical equations. ")
    
    st.divider()
    
    st.subheader("Dataset")

    st.markdown("In our study, we used two main datasets: MAWPS (A Math Word Problem Repository) for training and SVAMP (Simple Variation on Arithmetic Math Word Problems) for testing. The original research paper identified issues with widely used math word problem (MWP) benchmark datasets like ASDiv-A and MAWPS. Existing models performed well on these datasets, even when the “question” part was omitted during testing, indicating a reliance on cues unrelated to the actual math problem.")

    st.markdown("To address this, the researchers created the “SVAMP” dataset as a testing framework to evaluate a model’s proficiency in various aspects of mathematical word problem solving. SVAMP assesses sensitivity to questions, reasoning ability, and invariance to structural alterations. For example, it challenges models with variations like changing the direction of interactions or altering the sequence of events within a problem.")

    col1, col2 = st.columns(2)
    with col1:
        st.image("https://raw.githubusercontent.com/darkivist/Final-Project-Group1/ab1a9fe0168aa35d5d37017c58f0317fac0322b9/Code/Images/example-image.png")

    with col2:

        st.markdown("MAWPS:")
        st.markdown("- MAWPS was developed by researchers at Google and Microsoft to be used to test various NLP models curated to solve math word problems.")
        st.markdown("- Inconsistencies in the performance of models on this dataset because when the question component of the problem was removed unring test the model still preformed well.\n ")
        st.markdown("\t- Suggests potential reliance on cues unrelated to the actual mathematical concepts ")
    
        st.divider()

        st.subheader("Training and Testing Dataset")

        st.markdown("We trained our model on an augmented version of MAWPS with approximately 60,000 rows and used SVAMP with 1000 math word problems for testing. SVAMP includes scenarios focusing on subtraction, addition, division, and multiplication. The dataset provides information about the question, numbers, equations, and answers. The 'Numbers' column includes relevant numerical values for each problem, serving as inputs during data preprocessing. The 'Equation' column represents the target variable, aiding the evaluation of the model's ability to translate word problems accurately into a numeric format.")
    

# ----------- NLP Models -----------------------

with tab2:
    st.header("NLP Model Architecture")
    
    st.subheader("Original Attempts")
    
    st.text("GPT Model:")
    
    st.text("BERT Model:")
    
    st.subheader("T5 Model")
    
    st.text("Write description here:")
    
    

# ---------- Experimental Design and Hyperparameters ----------
with tab3:
    st.header("Experimental Design")
    
    st.text("Metrics and evaluations:")
    
    st.header("Hyperparameters")

    st.markdown("We employed Optuna for hyperparameter tuning, conducting multiple experiments to determine the optimal metrics for model evaluation. Initially, we explored minimizing loss and optimizing for exact matches and token-level accuracy between predicted and true answers in the validation set. However, tuning for token-level accuracy and exact answer match proved unsuccessful. The resulting model failed to produce correct validation predictions.")

    st.markdown("Subsequently, we focused on minimizing loss, and our tuner selected the following hyperparameter values: batch size - 64, epochs - 47, optimizer - Adam, and learning rate - 1e-4, resulting in a validation loss of 0.04. Unfortunately, the model produced with these parameters did not yield satisfactory results. After further experimentation, we settled on a batch size of 16, 200 epochs, optimizer Adam, and a learning rate of 1e-5, achieving an 80% correct prediction rate on our validation set.")



# ----------- Results and Limitation ------------

with tab4:
    col1, col2 =  st.columns([2,1])

    with col1:
        st.header("Results")

        st.divider()

        st.subheader("Metrics: Accuracy")

        



        st.markdown("We use accuracy as our primary metric for assessing mathematical output, employing the sympify function to verify equation equivalence. Additionally, we incorporate ROUGE scores to gauge the quality of generated equations by measuring overlap and similarity with reference equations.")

        st.markdown("Our test questions cover four arithmetic types: Subtraction (531), Addition (195), Common-Division (166), and Multiplication (108). This categorization enables a thorough assessment of our models' problem-solving skills across diverse mathematical contexts.")

        st.image("https://raw.githubusercontent.com/darkivist/Final-Project-Group1/516dd995db1e0c3b274e599c8a45fc4116630124/Code/Images/Accuracy_Flan_t5.png")


        st.divider()

        st.subheader("Metrics: ROUGE")

        st.markdown("The ROUGE scores (ROUGE-1: 0.605, ROUGE-2: 0.287, ROUGE-L: 0.605) assess the Language Model's (LLM) linguistic performance, measuring overlap and similarity with reference equations. These scores highlight the model's proficiency in reproducing unigrams, bigrams, and maintaining linguistic coherence")

        st.markdown(" A notable difference between ROUGE-2 and ROUGE-1 scores provides insights into the model's language generation capabilities, suggesting challenges in reproducing consecutive word sequences (bigrams) when ROUGE-2 is significantly lower than ROUGE-1."
                    )
        st.image("https://raw.githubusercontent.com/darkivist/Final-Project-Group1/516dd995db1e0c3b274e599c8a45fc4116630124/Code/Images/Rouge%20Score.png")

        st.divider()

        st.subheader("Flan T5 Base vs T5 small")

        st.markdown("Highlighting the crucial impact of model size and capacity on achieving higher accuracy rates, these results underscore Flan T5 Base's superior performance.")

        st.markdown("Flan T5 Base consistently outperforms T5 Small across arithmetic operations, with notable differences in Subtraction (19% vs. 10%), Addition (23% vs. 12%), Division (40% vs. 7%), and Multiplication (21% vs. 5%). In this comparison, Flan T5 Base achieves an overall accuracy of 23.6%, significantly outpacing T5 Small, which stands at 9.4%.")

        st.image("https://raw.githubusercontent.com/darkivist/Final-Project-Group1/516dd995db1e0c3b274e599c8a45fc4116630124/Code/Images/Accuracy_between_models.png")

        st.markdown("The noticeable differences in performance between Flan T5 Base and T5 Small can be attributed to various factors, with model size and capacity being the primary influence. As a larger model, Flan T5 Base has a higher parameter count and inherent complexity, enabling it to capture and generalize more intricate patterns within the data.")

        st.divider()

        st.markdown("Despite the significant variance in overall accuracy between Flan T5 Base and T5 Small, a more nuanced perspective emerges when considering ROUGE scores.")

        st.markdown("Flan T5 Base attains higher scores in both ROUGE-1 (0.605 vs. 0.537) and ROUGE-L (0.605 vs. 0.537), highlighting its superiority in unigram overlap and linguistic coherence. ")

        st.markdown("Nevertheless, the substantial decline in ROUGE-2 scores (0.287 vs. 0.144) indicates challenges for both models in accurately reproducing consecutive word sequences.")

        st.image("https://raw.githubusercontent.com/darkivist/Final-Project-Group1/516dd995db1e0c3b274e599c8a45fc4116630124/Code/Images/Rouge_Score_between_models.png")


    with col2:
        st.header("Limitation")


# ----------- Model Demo -------------------

with tab5:



    text = st.text_input("Type in a math problem", value=None, placeholder="Type here...")

    if st.button("Generate Answer"):
        # Tokenize and generate answer
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        output = model.generate(**inputs)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        st.write("Inputed math problem:", text)
        st.write("Generated Answer:", decoded_output)


# Add animation for when model is predicting



