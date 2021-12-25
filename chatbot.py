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
#Librería para introducir delays 
import time
#Librería para el manejo de codificadores y decodificadores (utf-8)
import codecs
#Librería para la gestión de texto a voz
import pyttsx3

model=load_model(os.getcwd() + '\chatbot_model')
lemmatizer = WordNetLemmatizer()
datosJson = json.loads(codecs.open('intents.json', 'r', 'utf-8').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

# Preprocesamiento 
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Técnica de Bag of Words (BOW) 
# Método que se utiliza en el procesado del lenguaje 
# para representar documentos ignorando el orden de las palabras.
def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return(np.array(bag))

def calc_pred(sentence, model):
    bag = bag_of_words(sentence, words)
    response = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(response) if r>ERROR_THRESHOLD]
    # ordenado por probabilidad
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def init_bot(msg):
    ints = calc_pred(msg, model)
    response = []
    #response = get_response(ints, datosJson)
    #Añadimos la probabilidad
    response.append(get_response(ints, datosJson))
    response.append(ints[0]['probability'])
    return response

# Inicializamos el sistema y lo preseentamos
userInput = 'Inicializando...'
engine = pyttsx3.init()
engine.setProperty('rate', 155) 

ia='ADR-IA'
print('\n\n%s'%(userInput))
engine.say(userInput)
engine.runAndWait()
response = init_bot(userInput)

print('\nBienvenido al Chatbot del TFG de Adrián Marzoa. Comienza un excitante encuentro con la inteligencia artificial %s, para salir escribe "Exit" o "Salir"'% (ia))
engine.say("Bienvenido al Chatbot del TFG de Adrián Marzoa. Comienza un excitante encuentro con la inteligencia artificial %s, para salir escribe, Exit, o, Salir " % (ia))
engine.runAndWait()
time.sleep(1)

print('\nIntroduce tu nombre para que podamos conversar')
engine.say("Introduce tu nombre para que podamos conversar")
engine.runAndWait()
time.sleep(1)

username = str(input("Nombre ->: ")).upper()
print('\n%s: Hola %s, ¿En que te puedo ayudar?'% (ia, username))
engine.say("Hola %s. ¿En que te puedo ayudar?"% (username))
engine.runAndWait()
time.sleep(1)

while userInput != 'Exit'  and userInput != 'Salir':
    userInput = str(input("%s: "% (username)))
    response = init_bot(userInput)
    # print("%s: "%(ia, response[0])) --- Sin añadir probabilidad a la respuesta
    # Añadimos la Probabilidad a la respuesta
    print("%s: %s -(Probabilidad)= %.2f%%" % (ia, response[0], float(response[1])*100))
    engine.say(response[0])
    engine.runAndWait()
