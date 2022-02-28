from vega_datasets import data
import string 
import streamlit as st
import altair as alt
import pandas as pd
import tkinter
import pickle
from textblob import TextBlob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from PIL import Image
image = Image.open('/Users/aggarwalpiush/Desktop/H4Hotel.jpeg')
image2 = Image.open('/Users/aggarwalpiush/Desktop/mercure.jpeg')

lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

def main():
    df = load_data()
    st.sidebar.title("About")
    
    st.sidebar.info("In this project we will choose the city and the aspects, by this input we will get the best hotel list of the following city according to their ranking")
    
    st.sidebar.title("Navigation")

    page = st.sidebar.radio("Choose a page", ["Homepage"])
    
    if page == "Homepage":
        st.header("Aspect based Sentiment Analysis")
        st.write("To analyse and perform a comparative study of the existing methods in the field of aspect-based sentiment analysis and to visualize the analysis in the form of a dashboard application.")
        st.write("Created by : Kunal")
        option = st.selectbox('Please select the city',('Duisburg', 'Düsseldorf', 'köln','Dortmund', 'Valencia', 'Milan'))
        st.write('You selected:', option)
        option = st.multiselect('Select aspects:', ['Location', 'Experience/Value', 'Service', 'Meal/Food','Room','Staff' ])
        st.write('Press submit')
        next = st.button('Review Analysis')
        if next:
            st.write("Hotel List")
            st.write("1. H4Hotel")

            st.image(image, caption='Google rating = 9.8 \n Model rating = 9.2')
            st.write("2. mercure")

            st.image(image2, caption='Google rating = 9.5 \n Model rating = 9')



    elif page == "Exploration":
        st.title("Data Exploration")
        user_input = st.text_input("Enter the file name", "/Users/aggarwalpiush/Desktop/dataframew2v_final.csv")
        df = pd.read_csv(user_input, encoding = 'latin-1')
        x_axis = st.selectbox("Choose a hotel name from the list ", df['Property name'].unique(),index = 3)
        visualize_data(df,x_axis)
    elif page == "Review Analysis":
        user_input = st.text_area("Enter the review", "Spent my birthday weekend at this lovely hotel --- area")
        analyse_data(user_input)

@st.cache
def load_data():
    df = pd.read_csv("/Users/aggarwalpiush/Desktop/dataframew2v_final.csv", encoding ='latin-1')
    return df

def visualize_data(df, x_axis):
    sns.barplot(x = 'Sentiment', y = "Aspect", data= df['Property Name']== x_axis)
    st.pyplot()
    
def analyse_data(data):
    column = ['Review', 'Aspect', 'Sentiment']
    reviews = list(filter(None, data.strip().split('.')))
    reviews = [basic_data_preprocessor(review) for review in reviews]
    reviews = [advanced_data_preprocessor(review) for review in reviews]
    cv_model = pickle.load(open("/Users/aggarwalpiush/Desktop/cv_model.sav", 'rb'))
    vector = cv_model.transform(reviews)
    aspect_model = pickle.load(open("/Users/aggarwalpiush/Desktop/lr_model_cv.sav", 'rb'))
    sent_model = pickle.load(open("/Users/aggarwalpiush/Desktop/lr_sent_model_cv.sav", 'rb'))
    as_pred = aspect_model.predict(vector)
    as_pred = [getAspectName(a) for a in as_pred]
    sent_pred_num = [getSentimentScore(text) for text in reviews]
    sent_pred = [getSentiment(s) for s in sent_pred_num]
    d = {'Review' : reviews, 'Aspect': as_pred, 'Sentiment': sent_pred}
    d_plot = {'Review': reviews, 'Aspect': as_pred, 'Sentiment': sent_pred_num}
    finaldf = pd.DataFrame(d, columns= column)
    st.write(finaldf)
    sns.barplot(x='Sentiment', y = "Aspect", data= finaldfPlot)
    st.pyplot()
    
def basic_data_preprocessor(text):
    tokens= word_tokenize(text)
    tokens= [w.lower() for w in tokens]
    tokens_clean = str.maketrans('','',string.punctuation)
    tokens_cleaned = [w.translate(tokens_clean) for w in tokens]
    clean_words = [word for word in tokens_cleaned if word.isalpha()]
    final_review = ''.join(clean_words)
    return final_review

def advanced_data_preprocessor(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words= [word for word in tokens if not word in stop_words]
    words=[spell.correction(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

def getVector(sent, model):
    temp = pd.DataFrame()
    for word in sent:
        try:
            word_vec = model[word]
            temp = temp.append(pd.Series(word_vec), ignore_index = True)
        except:
            pass
    return temp.mean()

def getAspectName(number):
    if number == 1:
        return 'Location'
    elif number == 0:
        return 'Experience/Value'
    elif number == 4:
        return 'Service'
    elif number == 2:
        return 'Meal/Food'
    elif number == 3:
        return 'Room'
    elif number == 5:
        return 'Staff'

def getSentiment(number):
    if number == 0:
        return 'Negative'
    elif number == 1:
        return 'Positive'

def getSentimentScore(text):
    score = TextBlob(text).sentiment.polarity
    if score >= 0.0:
        return 1;
    else:
        return 0;
    
if __name__ == "__main__":
    main()
        