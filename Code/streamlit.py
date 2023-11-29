import streamlit as st

# ---------- Titles and headers -----------
st.title("NLP Group 1: Translating Math Problems")
st.markdown("""___""")
st.header("Taking written math problems and using machine translation to compute the answer")
st.subheader("Paul Kelly, Carrie Magee, Jack McMorrow, Akshay Verma")
st.divider()

# ----------- Model Demo -------------------

st.write("Model Demo: Input a math problem here")

text = st.text_input("Type in a math problem", value=None, placeholder="Type here...")
st.write()
st.write("Inputed math problem:", text)

# Add animation for when model is predicting

# Code to predict the model

model_output = "*Need to load model*"
    
output = st.write(model_output)
st.write("Output of the trained model:", output)