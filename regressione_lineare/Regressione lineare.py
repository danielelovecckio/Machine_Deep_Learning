# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 10:41:28 2020

@author: daniele
"""
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib as mp

x = [1,2,3,4,5,6,7,8,9,10]      #features : dati in input
y = [37,31,42,28,32,34,39,45,43,48]  #target : dati in un output già noti

#istanziamo il modello
model = LinearRegression()

#strutturiamo i dati in forma bidimensionale, mettendoli in colonna
X = np.array(x).reshape(-1,1)

#vedo il grafico scatter per osservare come si distribuiscono i punti dei due array
mp.pyplot.scatter(X,y)
mp.pyplot.show()
#da qui posso capire che quando la x cresce, cresce anche la y, adesso vorremo osservare come si potrebbe comportare in futuro


#addestramento del modello
model.fit(X,y)

#[[11],[12],[13],[14]] equivale a dire array([11,12,13,14])
#ci stiamo chiedendo come si comporterebbe il modello se in input avessimo 11,12,13,14
print(model.predict([[11],[12],[13],[14]]))
#si vede dall'output che i valori delle y (target) saranno cresciuti cosi' come quelli delle x (features)
#ciò prende senso secondo l'andamento generico


#voglio mostrare su grafico ciò che è accaduto

#le features con cui abbiamo fatto il training
features = X

#le nuove x su cui vogliamo fare la previsione
x_per_previsione = np.array([[11],[12],[13],[14]])

#sequenza totale delle x che servirà per tracciare la retta di regressione
x_totali = np.concatenate((features,x_per_previsione))

#i valori target iniziali
target = y

#la previsione che eseguirà su tutte le x : la userò per la retta di regressione
previsione = model.predict(x_totali)

#punti usati per il training
mp.pyplot.scatter(features,target)

#retta di regressione prodotta
mp.pyplot.plot(x_totali,previsione,color = "orange") 
#dalla retta si vede che al crescere delle x crescono anceh le y e 
#la retta continua a salire

#questi (i quadrati più grandi) sono i valori che il modello ha predetto con le x non incluse nel training
mp.pyplot.scatter(x_per_previsione,previsione[-len(x_per_previsione):],marker = "s", s = 100, color = "blue")
mp.pyplot.show()


#----------------------------------------------------------------------------------------------------------------#



















