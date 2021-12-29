#Librería de procesamiento de Lenguaje Natural
import nltk
from nltk.stem import WordNetLemmatizer
#Librería para la creación de Redes Neuronales
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
#Librería para el manejo de archivos json
import json
#Librería para la utilización de objetos tipo pickle
import pickle
#Librería para el manejo de vectores y matrices multidimensionales
import numpy as np
#Librería para el manejo de codificadores y decodificadores (utf-8)
import codecs


lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?','¿', '!', '¡']

# Cargamos el json (utilizamos codificación utf-8)
intents = json.loads(codecs.open('intents.json', 'r', 'utf-8').read())

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

# Ahora lematizaremos cada palabra y eliminaremos las palabras duplicadas de la lista. 
# Lematizar es el proceso de convertir una palabra en su forma de lema 
# y luego crear un archivo pickle para almacenar los objetos de Python que usaremos al predecir.
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

# preparación para el entrenamiento de la red
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # Vector de palabras para utilizar la técnica de Bag of Words (bow) 
    bag = []
    # lista de tokens
    input_words = doc[0]
    # lematización del token
    input_words = [lemmatizer.lemmatize(word.lower()) for word in input_words]
    for w in words:
        if w in input_words:
            bag.append(1) 
        else:
             bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

training = np.array(training)

# creación del set de entrenamiento y de test: X - inputs, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])

# creación del modelo
model = Sequential()

# creación de las capas con sus neuronas y las funciones de activación
# la primera capa, la capa de entrada estará definida por input_dim, se crea con la adicción de la primera capa del modelo
# la capa oculta inicial que configuramos observamos que será de 128 neuronas y función de activación relu
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# Establecemos el Dropout. Este metodo ayuda a reducir el overfitting 
# ya que las neuronas cercanas suelen aprender patrones que se relacionan 
# y estas relaciones pueden llegar a formar un patron muy especifico con los datos de entrenamiento
# ver https://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf
model.add(Dropout(0.5))
# la segunda capa oculta observamos que será de 64 neuronas y función de activación relu
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# la capa de salida observamos que tendrá el tamaño de len(train_y[0]) y función de activación softmax
model.add(Dense(len(train_y[0]), activation='softmax'))

# Descenso de Gradiente (SGD) es un algoritmo de optimización muy utilizado en aprendizaje automático.
# SGD es un método general de minimización para cualquier función.
# Se carateriza por su uso extensivo en el campo de las redes neuronales. 
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenamos el modelo. epochs será el número de veces que se ejecutarán los algoritmos
trainRes = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
# Evaluamos el modelo.
scores= model.evaluate(np.array(train_x), np.array(train_y))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Guardamos el modelo para ser ejecutado.
model.save('chatbot_model', trainRes)

print("Modelo Creado y Entrenado")