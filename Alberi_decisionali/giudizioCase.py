# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:34:26 2020

@author: daniele
"""
import pandas as pd
dataframe = pd.read_csv('prezzi_case_class.csv')
print(dataframe)

X = dataframe[['vani','piano','ascensore','zona','prezzo']]
y = dataframe[['giudizio']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)      #inserendo random_state =0 ci assicuriamo che 
                                                                                #la divisione per i valori di training e test
                                                                                #sia sempre la stessa, dato che ogni volta questa 
                                                                                #è casuale
from sklearn.tree import DecisionTreeClassifier


model = DecisionTreeClassifier()        #costruisco il modello
model.fit(X_train,y_train)              #preparo il modello ad un ragionamento da seguire

previsione = model.predict(X_test)      #predico i giudizi che potrebbero avere le case che fanno parte di quella porzione di X
                                        #di test, in questo caso ho messo valori che non seguono un vero e proprio ragionamento coerente
                                        #di conseguenza non verranno azzeccate le y corrette (le y corrette sono le y_test)
print(X_train)                  
print(y_train)
print(X_test)
print(y_test)
print(previsione)

#posso verificare la poca accuratezza in questo modo
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,previsione))
print(model.get_depth())



#utilizzo della matrice di confusione che ci permetterà di vedere come si è comportato l'algoritmo
#sull'asse verticale vengono riprtati i target attesi
#sull'asse orizzontale vengono riportati i valori predetti
#i valori colorati sono quelli correttamente predetti 
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model,X_test,y_test)











