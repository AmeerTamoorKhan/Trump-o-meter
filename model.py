import tkinter as tk
import numpy as np
import tensorflow as tf
import pickle
WIDTH = 500
HEIGHT = 500
filename = "images/TrumpPic.png"
predicted_sentiment = None


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
    predict = tf.keras.preprocessing.sequence.pad_sequences(predict, maxlen=50)

    # Predict the Sentiment
    # 0 = Negative
    # 1 = Positive

    prediction = np.around(model.predict(np.array(predict)))
    print(prediction)
    return prediction


tokinizer, sentiment_model = load_model()


def analyze(text):
    global tokinizer, predicted_sentiment
    if text is "":
        pass
    else:
        predicted_sentiment = sentiment_analyzer(text, tokinizer, sentiment_model)
        sentiment_image()


def sentiment_image():
    global predicted_sentiment, filename

    if predicted_sentiment.astype(int)[0][0] == 1:
        pic['file'] = "images/NoTrump.png"
    elif predicted_sentiment.astype(int)[0][0] == 0:
        pic['file'] = "images/Trump.png"


root = tk.Tk()
root.title('Trump-O-Meter')

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg='white')
canvas.pack()


title = tk.Label(root, text="Is this by Trump?", font=('Courier New', 20, 'bold'))
title.place(relx=0.5, rely=0.05, relheight=0.1, anchor='n')

frame_upper = tk.Canvas(root, bg='#80c1ff', bd=5)
frame_upper.place(relx=0.5, rely=0.2, relwidth=0.8, relheight=0.1, anchor='n')

text = tk.Entry(frame_upper, font=('Courier New', 20, 'bold'), bd=2)
text.place(relwidth=0.8, relheight=1)

button = tk.Button(frame_upper, text='Analyze', command=lambda: analyze(text.get()))
button.place(relx=0.8, relwidth=0.2, relheight=1)


pic = tk.PhotoImage(file=filename)
pic_label = tk.Label(canvas, image=pic)
pic_label.place(relx=0.47, rely=0.4, anchor='n')

root.mainloop()

#comment = 'Will be interviewed tonight by @seanhannity at 9:00. Enjoy!@FoxNews'
#comment = 'Why would I allow the Debate Commission to change the rules for the second and third Debates when I easily won last time?'
#comment = "My sincere thanks for EAM Dr. Jaishankar's felicitations on the 71st Chinese National Day. Look forward to further promoting China-India relations."
#corrected comment = "Gamma is approaching hurricane status as it nears Mexico's Yucatan peninsula."
#comment = "Who next? Let Me know in the comments."
#comment = "Everything dies baby that's a fact. But maybe everything that dies someday comes back."
#comment = "What's going on everyone and welcome to the 2nd part of the chatbot with Python and TensorFlow tutorial series."






