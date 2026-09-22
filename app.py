import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from keras.utils import pad_sequences
import time

model = load_model('nextword_model_colab1.h5')
with open('tokenizer_colab1.pkl','rb') as file:
    tokenizer = pickle.load(file)

reverse_index = {idx:word for word, idx in tokenizer.word_index.items()}
max_len = 44

def generate_text(seed_text,num_words=10):
    text = seed_text
    yield text + " "
    for _ in range(num_words):
        seq = tokenizer.texts_to_sequences([text])[0]
        padded = pad_sequences([seq],maxlen=max_len,padding='pre')
        preds = model.predict(padded, verbose=0)
        pos = np.argmax(preds)
        next_word = reverse_index.get(pos," ")
        yield next_word + " "

        text = text + " " + next_word
        time.sleep(0.05) 
    # return text

st.title("Next Word Prediction Model")
st.subheader("Mini LLM with LSTM")

seed = st.text_input("Enter Initial Text: ", 'Hello')
num_words = st.slider("Number of words to generate",1,50,10)

if st.button("Generate"):
    result = generate_text(seed,num_words)
    st.write_stream(result)
    # st.write(result)
