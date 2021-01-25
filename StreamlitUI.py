import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf


def load_model():
    # Load the Tokenizer
    with open('utilities/trumpTokinizer.pk1', 'rb') as file:
        tokinizer = pickle.load(file)

    # Load the Model
    model = tf.keras.models.load_model('utilities/TweetByTrump.h5')

    return tokinizer, model


def sentiment_analyzer(comment, tokinizer, model):
    # Test sentences
    text = np.array([comment])
    predict = tokinizer.texts_to_sequences(np.array(text))
    #predict = tf.keras.preprocessing.sequence.pad_sequences(predict, maxlen=50)

    # Predict the Sentiment
    # 0 = Negative
    # 1 = Positive

    prediction = np.around(model.predict(np.array(predict)))
    print(prediction)
    return prediction


def analyze(text):
    global tokinizer, predicted_sentiment
    if text is "":
        pass
    else:
        predicted_sentiment = sentiment_analyzer(text, tokinizer, sentiment_model)
        return predicted_sentiment


def default():
    st.header('Working Demonstration:')
    st.video('images/demo.mp4')
    st.header('How is Works:')
    st.markdown('''
            <p>Trump-o-meter is a fun project designed to understand the dynamics of Trump's wording in his tweeets. The
             project is trained using LSTM-RNN and the model is trained on 40,000 tweets, more or less half are by the
             Trump and the remaining are by other people. With the limited dataset th project has the accuracy of almost
            85%, which seems pretty ok.</p>
            <h3><strong>#LSTM</strong>  <strong>#RNN</strong>  <strong>#python</strong>  <strong>#trump</strong> 
            <strong>#machinelearning</strong></h3>
            ''', unsafe_allow_html=True)


tweets = [['Be sure to tune in and watch Donald Trump on Late Night with David Letterman as he presents the Top'
           ' Ten List'
           ' tonight!', 'Yes'],
          ['I dived many times for the ball. Managed to save 50% The rest go out of bounds', 'No'],
          ['Donald Trump will be appearing on The View tomorrow morning to discuss Celebrity Apprentice and his new '
           'book'
           ' Think Like A Champion!', 'Yes'],
          ['my whole body feels itchy and like its on fire', 'No'],
          ['Donald Trump reads Top Ten Financial Tips on Late Show with David Letterman: http://tinyurl.com/ooafwn - '
           'Very funny!', 'Yes']]
df = pd.DataFrame(tweets, columns=['Tweets', 'isItTrump'])

st.set_page_config('Trump-o-meter')
cols_title = st.beta_columns((1, 0.1, 3, 1, 1))
cols_title[0].image('images/TrumpPic.png', width=150)
cols_title[2].title("Trump-o-meter")

intro_btn = st.sidebar.radio('Select', ('Introduction', 'Start'))

st.sidebar.markdown('''<h3>Created By: Ameer Tamoor Khan</h3>
                    <h4>Github : <a href="https://github.com/AmeerTamoorKhan" target="_blank">Click Here </a></h4> 
                    <h4>Email: drop-in@atkhan.info</h4> ''', unsafe_allow_html=True)

if intro_btn == 'Start':
    st.header('Enter Tweet:')
    tweet = st.text_input('')

    cols_pics = st.beta_columns((1, 1, 1))
    with cols_pics[1]:
        img = st.empty()
        # img.image('images/TrumpPic.png', width=200)

    cols_btn = st.beta_columns((0.1, 0.1))
    with cols_btn[0]:
        check = st.button('Check')

    if check:
        tokinizer, sentiment_model = load_model()
        result = analyze(tweet)
        if result == 0:
            img.image('images/Trump.png', width=250)
        else:
            img.image('images/NoTrump.png', width=250)
    st.header('Example Tweets:')
    st.table(df)
elif intro_btn == 'Introduction':
    st.empty()
    default()




