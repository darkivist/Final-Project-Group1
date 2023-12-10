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

output_dir_ans = "darkivist/t5_math_problems"
tokenizer_dir_ans = "darkivist/t5_math_problems"

tokenizer_ans = T5Tokenizer.from_pretrained(tokenizer_dir_ans)



def load_model_ans():
    model = T5ForConditionalGeneration.from_pretrained(output_dir_ans)
    return model

model_ans = load_model_ans()


# ---------- Titles and headers -----------
st.title("NLP Group 1: Translating Math Problems")
st.subheader("Paul Kelly, Carrie Magee, Jack McMorrow, Akshay Verma")


tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introduction & Dataset", "NLP Model", "Experimental Design and Hyperparameters"  , "Results & Limitaion", "Model Demo"])



with tab1:

    
    st.subheader("Introduction")
    

    st.markdown("Language complexity gets tricky, especially when mixing math into word problems. Our group took on this challenge using transformers to simplify the complexities in math and languages")
    st.markdown("Math word problems, those short stories with real-world scenarios, need more than just basic math skills. You've got to understand the context, sentence structure, and word connections. Solving them means figuring out the problem, picking out the important info, and turning it into solvable math. Computers find this tricky due to ambiguity and understanding context.")

    st.image("https://raw.githubusercontent.com/darkivist/Final-Project-Group1/05a63502c78993eb5725fc1a832223d9f34c3e9f/Code/Images/1_6qu0JSsBvKsyME8Wd_xF6Q.png")
    st.divider()

    st.subheader("Goals of the project")
    
    st.markdown(" In our attempt to solve this problem, the multimodal capacities of transformers emerged as valuable assets. The goal of our project is to solve linguistic challenges with computational solutions, more specifically use the power of deep learning to convert word problems into solvable mathematical equations. ")
    
    st.divider()
    
    st.subheader("Dataset")

    st.markdown("In our study, we used two main datasets: MAWPS (A Math Word Problem Repository) for training and SVAMP (Simple Variation on Arithmetic Math Word Problems) for testing. The original research paper identified issues with widely used math word problem (MWP) benchmark datasets like ASDiv-A and MAWPS. Existing models performed well on these datasets, even when the “question” part was omitted during testing, indicating a reliance on cues unrelated to the actual math problem.")

    st.markdown("To address this, the researchers created the “SVAMP” dataset as a testing framework to evaluate a model’s proficiency in various aspects of mathematical word problem solving. SVAMP assesses sensitivity to questions, reasoning ability, and invariance to structural alterations. For example, it challenges models with variations like changing the direction of interactions or altering the sequence of events within a problem.")

    col1, col2 = st.columns(2,  gap="medium")
    with col1:
        st.image("https://raw.githubusercontent.com/darkivist/Final-Project-Group1/ab1a9fe0168aa35d5d37017c58f0317fac0322b9/Code/Images/example-image.png")

    with col2:

        st.markdown("MAWPS:")
        st.markdown("- MAWPS was developed by researchers at Google and Microsoft to be used to test various NLP models curated to solve math word problems.")
        st.markdown("- Inconsistencies in the performance of models on this dataset because when the question component of the problem was removed unring test the model still preformed well.\n ")
        st.markdown("\t- Suggests potential reliance on cues unrelated to the actual mathematical concepts ")
        
        st.markdown("SVAMP:")
        st.markdown("- Researchers developed SVAMP in response to the shortcomings of the MAWPS dataset.")
        st.markdown("- This testing dataset used to assess models’ proficiency in various aspects of math word problem solving by including MWP that vary in question sensitivity, reasoning ability, and structural invariance ")
        
        st.divider()
        
        st.markdown("Aspect of Language Structure to Consider:")
        st.markdown("- Question Sensitivity: variations in SVAMP check if the model’s answer depends on the structure of the question.")
        st.markdown("- Reasoning Ability: variations ensure the model has learned to correctly determine a change in reasoning arising from changes in the problem text.")
        st.markdown("- Structural Invariance: ensures that the model remains invariant to superficial changes in the problem text (i.e., changes that do not alter the answer or reasoning)")
        
        st.divider()
        
    st.subheader("Training and Testing Dataset")

    st.markdown("We trained our model on an augmented version of MAWPS with approximately 60,000 rows and used SVAMP with 1000 math word problems for testing. SVAMP includes scenarios focusing on subtraction, addition, division, and multiplication. The dataset provides information about the question, numbers, equations, and answers. The 'Numbers' column includes relevant numerical values for each problem, serving as inputs during data preprocessing. The 'Equation' column represents the target variable, aiding the evaluation of the model's ability to translate word problems accurately into a numeric format.")
    

# ----------- NLP Models -----------------------

with tab2:
    col1, col2 = st.columns([1,2], gap='medium')
    with col1:
        st.header("NLP Model Architecture")

        st.subheader("Original Attempts")

        st.markdown("We explored various deep learning approaches in order to achieve our goal of translating math word problems into a numeric output. The first approach attempted to use a GPT-3 type transformer due to its flexibility in various NLP tasks like language translation, summarization, and generation. Though the model showed extremely low loss after training, it ultimately was unable to translate math word problems into numeric equations thus, leaving the team to explore other sequence-to-sequence focused techniques.")
        st.markdown("GPT Model: The GPT-3 type transformer follows the transformer architecture, known for its attention mechanisms. It excels in capturing contextual information across sequences. Cross-entropy loss was employed, commonly used in language modeling tasks, with the expectation that it would guide the model to generate accurate numeric equations for math word problems.Despite achieving low training loss, the model struggled to make accurate predictions, leading the team to reevaluate the chosen approach.")

        st.markdown("BERT Model: BERT, or Bidirectional Encoder Representations from Transformers, is a transformer-based model designed for bidirectional context understanding. It is particularly effective in capturing dependencies in both directions. : Similar to GPT-3, cross-entropy loss was applied, aiming to guide the model in understanding the sequential information in math word problems. BERT, too, faced difficulties in accurately translating math word problems into numeric equations, prompting a reassessment of the chosen architecture.")
    with col2:
        st.image("https://raw.githubusercontent.com/darkivist/Final-Project-Group1/a0b95c8a1e17944b1df85aae7ca56098e40c7805/Code/Images/llm_tree.jpg")
    st.subheader(" Flan T5 Model")
    
    col1, col2 = st.columns(2, gap='medium')
    with col1:
        st.image("https://raw.githubusercontent.com/darkivist/Final-Project-Group1/235075974729029e89078a4dd265ebeaacab06b4/Code/Images/flan2_architecture.jpg")
    with col2:
        st.markdown("Flan T5 Base, an extension of Google's T5 architecture, specializes in language-related tasks, with a particular focus on mathematical language understanding. Boasting 245 million parameters, this variant offers versatility for natural language processing (NLP) tasks, treating them uniformly as text-to-text problems. Through fine-tuning on the MAWPS augmented dataset, Flan T5 Base becomes adept at converting mathematical word problems to equations, capitalizing on the models adaptability inherited from the T5 architecture.")
        st.markdown("The model's efficacy in addressing the nuances of mathematical language can be attributed to its robustness and adaptability. Like its predecessor T5, Flan T5 Base utilizes standard language modeling loss, such as cross-entropy loss, during training. This ensures the generation of accurate equations that faithfully represent the mathematical relationships described in word problems. The fine-tuning process is crucial, involving experimentation with hyperparameters to attain an optimal configuration for the specific task at hand, thereby enhancing the model's performance in converting MWPs to equations")
        st.markdown("In summary, Flan T5 Base, with its foundation in the T5 architecture, proves to be a powerful tool for mathematical language understanding. Through fine-tuning on the MAWPS augmented dataset and a meticulous optimization process, the model excels in converting complex mathematical word problems into accurate equations, showcasing its adaptability and robustness in handling diverse NLP challenges.")
    

# ---------- Experimental Design and Hyperparameters ----------
with tab3:
    st.header("Experimental Design")
    
    st.markdown("Before discovering our augmented dataset, we initially worked with a dataset that had distinct Numbers and Questions columns. To facilitate model training, we developed a function that utilized regular expressions to replace number placeholders within the questions. Specifically, for the T5-small model, we implemented a class that created customized training and validation datasets. In this setup, the processed Question (with real numbers) and Equation fields served as inputs, while the Answer field served as the output. Our Seq2Seq trainer was then instantiated with the specified training arguments.")
    st.markdown("In the context of equation generation models, we transitioned to using the Flan T5 Base model after experimenting with T5-small. The setup for Flan T5 Base mirrors that of T5-small, maintaining consistency in the approach. This transition allowed us to leverage the enhanced capabilities of Flan T5 Base for our specific task of converting mathematical word problems to equations, ensuring a seamless integration into our existing framework.")
    st.header("Hyperparameters")

    st.markdown("We employed Optuna for hyperparameter tuning, conducting multiple experiments to determine the optimal metrics for model evaluation. Initially, we explored minimizing loss and optimizing for exact matches and token-level accuracy between predicted and true answers in the validation set. However, tuning for token-level accuracy and exact answer match proved unsuccessful. The resulting model failed to produce correct validation predictions.")

    st.markdown("Subsequently, we focused on minimizing loss, and our tuner selected the following hyperparameter values: batch size - 64, epochs - 47, optimizer - Adam, and learning rate - 1e-4, resulting in a validation loss of 0.04. Unfortunately, the model produced with these parameters did not yield satisfactory results. After further experimentation, we settled on a batch size of 16, 200 epochs, optimizer Adam, and a learning rate of 1e-5, achieving an 80% correct prediction rate on our validation set.")

    st.image("https://raw.githubusercontent.com/darkivist/Final-Project-Group1/26716026dff9598f704d31aa9660e5e43d47a9d0/Code/Images/eval_loss.png", caption="Eval Loss for Equations")


# ----------- Results and Limitation ------------

    with tab4:
        st.header("Results")
        col1, col2 =  st.columns(2,  gap="medium")
        

        with col1:
            

            

            st.subheader("Metrics: Accuracy")

            



            st.markdown("We use accuracy as our primary metric for assessing mathematical output, employing the sympify function to verify equation equivalence. Additionally, we incorporate ROUGE scores to gauge the quality of generated equations by measuring overlap and similarity with reference equations.")

            st.markdown("Our test questions cover four arithmetic types: Subtraction (531), Addition (195), Common-Division (166), and Multiplication (108). This categorization enables a thorough assessment of our models' problem-solving skills across diverse mathematical contexts.")

            st.image("https://raw.githubusercontent.com/darkivist/Final-Project-Group1/516dd995db1e0c3b274e599c8a45fc4116630124/Code/Images/Accuracy_Flan_t5.png", width=800)

            st.divider()

            st.subheader("Flan T5 Base vs T5 small")

            st.markdown("Highlighting the crucial impact of model size and capacity on achieving higher accuracy rates, these results underscore Flan T5 Base's superior performance.")

            st.markdown("Flan T5 Base consistently outperforms T5 Small across arithmetic operations, with notable differences in Subtraction (19% vs. 10%), Addition (23% vs. 12%), Division (40% vs. 7%), and Multiplication (21% vs. 5%). In this comparison, Flan T5 Base achieves an overall accuracy of 23.6%, significantly outpacing T5 Small, which stands at 9.4%.")

            st.image("https://raw.githubusercontent.com/darkivist/Final-Project-Group1/516dd995db1e0c3b274e599c8a45fc4116630124/Code/Images/Accuracy_between_models.png", width=800)


        with col2:


            st.subheader("Metrics: ROUGE")

            st.markdown("The ROUGE scores (ROUGE-1: 0.605, ROUGE-2: 0.287, ROUGE-L: 0.605) assess the Language Model's (LLM) linguistic performance, measuring overlap and similarity with reference equations. These scores highlight the model's proficiency in reproducing unigrams, bigrams, and maintaining linguistic coherence")

            st.markdown(" A notable difference between ROUGE-2 and ROUGE-1 scores provides insights into the model's language generation capabilities, suggesting challenges in reproducing consecutive word sequences (bigrams) when ROUGE-2 is significantly lower than ROUGE-1." )
            st.image("https://raw.githubusercontent.com/darkivist/Final-Project-Group1/516dd995db1e0c3b274e599c8a45fc4116630124/Code/Images/Rouge%20Score.png",  width=750)



            st.divider()

            st.markdown("Despite the significant variance in overall accuracy between Flan T5 Base and T5 Small, a more nuanced perspective emerges when considering ROUGE scores.")

            st.markdown("Flan T5 Base attains higher scores in both ROUGE-1 (0.605 vs. 0.537) and ROUGE-L (0.605 vs. 0.537), highlighting its superiority in unigram overlap and linguistic coherence. ")

            st.markdown("Nevertheless, the substantial decline in ROUGE-2 scores (0.287 vs. 0.144) indicates challenges for both models in accurately reproducing consecutive word sequences.")

            st.image("https://raw.githubusercontent.com/darkivist/Final-Project-Group1/516dd995db1e0c3b274e599c8a45fc4116630124/Code/Images/Rouge_Score_between_models.png", width=800)

        st.markdown("The noticeable differences in performance between Flan T5 Base and T5 Small can be attributed to various factors, with model size and capacity being the primary influence. As a larger model, Flan T5 Base has a higher parameter count and inherent complexity, enabling it to capture and generalize more intricate patterns within the data.")

        st.divider()

        st.header("Accuracy difference between numerical answers generation and equation generation")

        st.markdown("Our equation-answering model demonstrates commendable accuracy, revealing a significant performance gap when compared to a model exclusively focused on numerical responses. Notably, the numerical model encounters challenges, with accuracy percentages varying across operations: Subtraction at 57%, Addition at 12%, Common-Division at 25%, and Multiplication at 5%. This disparity suggests potential complexities in the model's numerical reasoning, prompting a need for further investigation to enhance its precision in predicting correct numerical solutions.")

        st.image("")

        st.divider()
        st.header('Limitation')

        st.markdown("Despite the promising features of the Flan T5 Base model, it's essential to acknowledge its limitations. The model's accuracy falls short when faced with more complex mathematical word problems, demonstrating a noticeable struggle in handling higher-level mathematical concepts. While adept at basic arithmetic and straightforward problem-solving, its performance tends to plateau, making it less reliable for intricate or advanced mathematical scenarios. This limitation underscores the importance of continued research and development to enhance the model's capabilities and extend its applicability to a broader range of mathematical complexities. Acknowledging these constraints provides a transparent understanding of the model's current limitations and serves as a foundation for future improvements and advancements in mathematical language understanding models.")



# ----------- Model Demo -------------------

with tab5:

    st.header("Type in a Math Word Problem")

    text = st.text_input("", value=None, placeholder="Type here...")

    if st.button("Generate Answer"):
        # Tokenize and generate equation
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        output = model.generate(**inputs)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

        # Tokenize and generate answer
        inputs_ans = tokenizer_ans(text, return_tensors="pt", max_length=512, truncation=True)
        output_ans = model_ans.generate(**inputs_ans)
        decoded_output_ans = tokenizer.decode(output_ans[0], skip_special_tokens=True)

        # Display results in a box with big font
        st.subheader("Generated Results:")

        # Use Streamlit columns to display side by side
        col1, col2 = st.columns(2)

        with col1:
            st.info("Generated Equation:")
            st.write(decoded_output, key="equation_output")

        with col2:
            st.success("Generated Answer:")
            st.write(decoded_output_ans, key="answer_output")

# Add animation for when model is predicting



