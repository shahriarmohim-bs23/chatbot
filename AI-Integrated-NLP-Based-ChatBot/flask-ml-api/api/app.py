# libraries
import wikipedia
import random
import pyttsx3
import numpy as np
import pickle
import json
from flask import Flask, render_template, request
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# chat initialization
model = load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app = Flask(__name__)


def text_to_speech(text):
    engine = pyttsx3.init('dummy')
    engine.setProperty('rate', 150)  # Setting up voice rate
    engine.setProperty('volume', 80)  # Setting up volume level  between 0 and 1
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  # Change voices: 0 for male and 1 for female

    engine.say(text)
    engine.runAndWait()
    engine.stop()


@app.route('/')
@app.route('/chatbot')
def home():
    return render_template('MasumTheBot.html',
                           title='MasumTheBot')


@app.route('/get')
def get_bot_response():
    userText = request.args.get('msg')
    # print(userText)
    return chatbot_response(str(userText))


def remove_substring_from_string(s, substr):
    for i in range(len(s) - len(substr) + 1):
        if s[i:i + len(substr)] == substr:
            break
    else:
        return s

    return s[:i] + s[i + len(substr):]


def chatbot_response(msg):
    # msg = request.form["msg"]
    special_word = "more info"
    res = ""

    if special_word in msg:
        try:
            x = remove_substring_from_string(msg, special_word)
            res = wikipedia.summary(x, sentences=1)
            print(res)
        except:
            res = "My mood isn't good now, there-by could not understand you. Ask again, please."
    else:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
    text_to_speech(res)
    return res


# chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0")
    