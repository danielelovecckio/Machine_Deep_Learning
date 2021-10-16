#
"""
Created on Tue Dec 22 17:46:01 2020

@author: daniele
"""
import numpy as np
import matplotlib.pyplot as plt

#Sequential mi costruisce la rete nurale
from keras.models import Sequential

#dense è un metodo che mi permette di collegari i neuroni del livello precedente con i nodi del livello attuale
from keras.layers import Dense


#prendo dal orgetto mnist la funzione load_mnist
from mnist import load_mnist

x_train, x_test, y_train, y_test = load_mnist (path="C:/Users/daniele/Desktop/Reti_Neuralipy/MNIST")


#stampa del primo elemento di x_train
plt.imshow(x_train[0].reshape([28,28]),cmap = "gray")
plt.axis('off') # rimuoviamo i valori sulle assi 
print("La cifra nell'immagine è un %d" % y_train[0])

#Pre_processing
#Normalizzazione
#si divide ogni pixel con il pixel massimo che è 255
x_train=x_train/255
x_test=x_test/255



#siccome abbiamo 10 categorie utiliziamo la funzione di mkeras per categorizare

from keras.utils import to_categorical
num_class=10
y_train_nuovo=to_categorical(y_train,num_class)
y_test_nuovo=to_categorical(y_test,num_class)




#Creo il modello
model=Sequential()
#il primo livello nascosto avra 512 noti
model.add(Dense(512,input_dim=x_train.shape[1],activation='relu'))
#aggiungo un altro livello
model.add(Dense(256,activation='relu'))
#terzo livello nascosto
model.add(Dense(128, activation='relu'))
#ultimo livello per output che avra il numero di nodi=num_class
#con la multiclasse come attivazione si passa softmax
model.add(Dense(num_class,activation='softmax'))

#analiziamo
model.summary()


#preparazione fase addrestramento
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

model.fit(x_train,y_train_nuovo, epochs=20)


#verifichiamo sul set di test
print('\n\n\n set test')
model.evaluate(x_test,y_test_nuovo)




