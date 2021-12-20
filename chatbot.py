#Librerías de procesamiento de Lenguaje Natural
import nltk
from nltk.stem import WordNetLemmatizer
#Librería para la creación de Redes Neuronales
import tensorflow as tf
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.keras.models import load_model
#Librería para el manejo de archivos json
import json
#Librería para la utilización de objetos tipo pickle
import pickle
#Librería para el manejo de vectores y matrices multidimensionales
import numpy as np
#Librería para la generación de números pseudoaleatorios
import  random
#Librería de funcionalidad dependientes del sistema operativo (paths y otras)
import os

model=load_model(os.getcwd() + '\chatbot_model')
lemmatizer = WordNetLemmatizer()
datosJson = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

# Preprocesamiento 
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Técnica de Bag of Words (bow) 
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return(np.array(bag))

def calcula_pred(sentence, model):
    bag = bow(sentence, words)
    response = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(response) if r>ERROR_THRESHOLD]
    # ordenado por probabilidad
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def initBot(msg):
    ints = calcula_pred(msg, model)
    response = getResponse(ints, datosJson)
    return response

userInput = ''
print('Bienvenido al Chat UOC TFG, para salir escribe "Exit"')

while userInput != 'Exit':
    userInput = str(input("USUARIO: "))
    response = initBot(userInput)
    print('BOT UOC:' + response)