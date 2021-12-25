#Librerías de procesamiento de Lenguaje Natural
import nltk
from nltk.stem import WordNetLemmatizer
#Librería para la creación de Redes Neuronales
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
#Librería para el manejo de archivos json
import json
#Librería para la utilización de objetos tipo pickle
import pickle
#Librería para el manejo de vectores y matrices multidimensionales
import numpy as np
import unicodedata
import codecs


lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = codecs.open('intents.json', 'r', 'utf-8').read()
intents = json.loads(data_file)

print(intents)



# intents: Conversaciones Tipo (Intenciones del Usuario)
# inputs: Posibles inputs de interacción del usuario (Inputs del Usuario)
for intent in intents['intents']:
    for input in intent['inputs']:

        # tokenizamos 
        w = nltk.word_tokenize(input)
        words.extend(w)
        # agrego a la matriz de documentos
        documents.append((w, intent['tag']))

        # agregamos clases a la lista de clases
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
