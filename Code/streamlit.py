import streamlit as st

import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

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
st.markdown("""___""")
st.header("Taking written math problems and using machine translation to compute the answer")
st.subheader("Paul Kelly, Carrie Magee, Jack McMorrow, Akshay Verma")
st.divider()





# ----------- Model Evaluation ------------




# ----------- Model Demo -------------------

st.write("Model Demo: Input a math problem here")

text = st.text_input("Type in a math problem", value=None, placeholder="Type here...")


if st.button("Generate Answer"):
    # Tokenize and generate answer
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    output = model.generate(**inputs)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write("Generated Answer:", decoded_output)

st.write()
st.write("Inputed math problem:", text)

# Add animation for when model is predicting


# Code to predict the model

model_output = "*Need to load model*"
    
output = st.write(model_output)
st.write("Output of the trained model:", output)

