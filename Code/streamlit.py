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
    st.markdown("""___""")
    
    # Uploading stock image
    image = Image.open("stock-image.jpeg")
    st.image(image)

    
    st.divider()
    
    st.markdown("Ambiguity, context-dependent information, and inferential reasoning in math word problems poses a unique challenge to transformers and other NLP models")

    st.subheader("Goals of the project:")
    
    st.markdown("Goal of project is to translate math word problems (MWP) into numeric equations using transformers ")
    
    st.divider()
    
    st.subheader("Dataset")

    st.markdown("MAWPS (Math Word Problem Repository) used for training")
    st.markdown("SVAMP (Simple Variation on Arithmetic Math Word Problems) used for testing")
    
    st.markdown("MAWPS:")
    st.markdown("- MAWPS was developed by researchers at Google and Microsoft to be used to test various NLP models curated to solve math word problems.")
    st.markdown("- Inconsistencies in the performance of models on this dataset because when the question component of the problem was removed unring test the model still preformed well.\n ")
    st.markdown("\t- Suggests potential reliance on cues unrelated to the actual mathematical concepts ")
    
    #st.image("example-image.png")
    

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




# ----------- Results and Limitation ------------

with tab4:
    col1, col2 =  st.columns([2,1])

    with col1:
        st.header("Results")
        st.image("https://raw.githubusercontent.com/darkivist/Final-Project-Group1/87755f1ff1f9da1922d14ffd03f7fb5e6ee8ce7e/Code/Images/bar-graph.png")
        st.image("https://raw.githubusercontent.com/darkivist/Final-Project-Group1/8926883058a453f8b355205755cd1632006bca49/Code/Images/Rouge%20Score.png")

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



