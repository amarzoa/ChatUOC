
import nltk, json, random, pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.engine.sequential import Sequential

import pickle
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

model=load_model(os.getcwd() + '\chatbot_model')
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

# preprocesamiento 
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def calcula_pred(sentence, model):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # ordenado por probabilidad
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getRespuesta(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def inicia(msg):
    ints = calcula_pred(msg, model)
    res = getRespuesta(ints, intents)
    return res

usuario = ''
print('Bienvenido al Chat UOC TFG, para salir escribe "Exit"')

while usuario != 'Exit':
    usuario = str(input(""))
    res = inicia(usuario)
    print('BOT UOC:' + res)