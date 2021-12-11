#ljdfsñlgjsdflgjsdfgsdgsdfgs
import pickle
import nltk 
from nltk.stem import WordNetLemmatizer
import json
import numpy
import tensorflow
import numpy as np  

from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout 
from tensorflow.python.keras.optimizers import SGD  

randomwords=[]  
classes = []  
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()  
words=""
pik=pickle

intents = json.loads(data_file)

lemmatizer = WordNetLemmatizer()

for intent in intents['intents']:   
    for pattern in intent['patterns']:   
        #tokenize each word   
        w = nltk.word_tokenize(pattern)   
        w.extend(w) #add documents in the corpus   
        documents.append((w, intent['tag']))   # add to our classes list   
        if intent['tag'] not in classes:   
            classes.append(intent['tag'])

# lemmatize, lower each word and remove duplicates  
w = [lemmatizer.lemmatize(w.lower())  for w in words if w not in ignore_words]   
w = sorted(list(set(words)))# sort classes  
classes = sorted(list(set(classes))) # documents = combination between patterns and intents  
print(len(documents), "documents")# classes = intents  
print(len(classes), "classes", classes)# words = all words, vocabulary  
print(len(w), "unique lemmatized words", words)  
pickle.dump(w,open('words.pkl','wb'))  
pickle.dump(classes,open('classes.pkl','wb'))

print("Bot: Hola, Mi nombre es BOTUOC")
