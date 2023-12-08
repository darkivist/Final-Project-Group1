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


tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introduction & Dataset", "NLP Model", "Experimental Design and Hyperparameters"  , "Results & Limitaion", "Model Demo"])



with tab1:

    st.markdown("""___""")
    
    # Uploading stock image
    image = Image.open("stock-image.jpeg")
    st.image(image)
    st.divider() 
    
    st.header("Taking written math problems and using machine translation to compute the answer")
    st.subheader("Paul Kelly, Carrie Magee, Jack McMorrow, Akshay Verma")
    
    st.divider()

    st.text("""The goal of this project is to utlized a transformer to be able to solve word problem statements that
            are commonly used in classrooms. ......""")
    
    st.divider()
    
    st.subheader("Dataset")
    
    st.image("example-image.png")
    st.text(""" ***Description of dataset*** """)
    
    example_df = pd.DataFrame({
        
    })

# ----------- NLP Models -----------------------

with tab2:
    st.header("NLP Model Architecture")
    
    

# ---------- Experimental Design and Hyperparameters ----------
with tab3:
    st.header("Experimental Design")
    
    




# ----------- Results and Limitation ------------

with tab4:
    col1, col2 =  st.columns([2,1])

    with col1:
        st.header("Results")
        st.image(r"C:\Users\Akshay\OneDrive\Desktop\RA\Final-Project-Group1\Code\Rouge Score.png")


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



