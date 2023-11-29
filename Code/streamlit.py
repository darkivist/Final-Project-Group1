# Imports
import streamlit as st
import time

# ---------- Define Functions -------------
def simulate_loading_process():
    time.sleep(3)
    
def progress_bar_test(text):
    for i in range(1, 101):
        time.sleep(0.1)
        progress_bar.progress(i)
    
    return f"Inputed Text: {text}"

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
st.write()

# Add animation for when model is predicting
if text:
    progress_bar = st.progress(0)
    
    prediction = progress_bar_test(text)
    
    progress_bar.empty()
    st.success(prediction)
    
# Consider adding progess bar

# Code to predict the model

model_output = "*Need to load model*"
    
output = st.write(model_output)
st.write("Output of the trained model:", None)

