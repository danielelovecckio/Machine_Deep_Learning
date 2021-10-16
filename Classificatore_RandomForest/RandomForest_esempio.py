# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:23:08 2020

@author: daniele
"""
#ensamble è una libreria che contiene diversi classificatori più generali come RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

import pandas as pd

dataframe = pd.read_csv('prezzi_case_class.csv')
print(dataframe)

X = dataframe[['vani','piano','ascensore','zona','prezzo']]
y = dataframe[['giudizio']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)     
                                                                               

#ADDESTRAMENTO DEL MODELLO
model.fit(X_train,y_train)
#Eseguo LA PREVISIONE
previsione = model.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix

#MISURO L'ACCURATEZZA
print(accuracy_score(y_test,previsione))
#VEDO IL COMPORTAMENTO CON LA MATRICE DI CONFUSIONE
plot_confusion_matrix(model,X_test,y_test)