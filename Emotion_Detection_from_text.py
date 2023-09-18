import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Define the preprocess function
ps = PorterStemmer()

def preprocess(line):
    review = re.sub('[^a-zA-Z]', ' ', line)      #leave only characters from a to z
    review = review.lower()                      #lower the text
    review = review.split()                      #turn string into a list of words
    
    # Apply stemming
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]  #delete stop words
    
    # Turn the list into a sentence
    return " ".join(review)

# Load the label encoder
label_encoder_file_path = 'pages/encoder.pkl'
label_encoder = pickle.load(open(label_encoder_file_path, 'rb'))

# Load the CountVectorizer
cv_file_path = 'pages/CountVectorizer.pkl'
cv = pickle.load(open(cv_file_path, 'rb'))

# Load the Keras model
model_file_path = 'pages/my_model.h5'
model = load_model(model_file_path)

# Streamlit app title
st.title('Emotion Detection from Text')

# Input text area for user input
input_text = st.text_area('Enter text for emotion detection:', 'I feel happy')

# Button to trigger emotion detection
if st.button('Detect Emotion'):
    # Preprocess the input text
    preprocessed_text = preprocess(input_text)
    
    # Vectorize the preprocessed text
    text_vector = cv.transform([preprocessed_text]).toarray()
    
    # Predict the emotion
    pred = model.predict(text_vector)
    predicted_emotion_index = np.argmax(pred, axis=1)[0]
    
    # Decode the predicted emotion label using the label encoder
    predicted_emotion = label_encoder.inverse_transform([predicted_emotion_index])[0]
    
    st.subheader('Emotion Detected:')
    st.write(predicted_emotion)
