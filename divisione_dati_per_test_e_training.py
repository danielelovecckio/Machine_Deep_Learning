# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:27:39 2020

@author: daniele
"""
#DIVISIONE DI UN DATASET PER UNA PARTE DI TRAINING E UNA PARTE DI TEST

from sklearn.model_selection import train_test_split  #funzione che mi permetterà di fare la divisione del dataset
from sklearn.datasets import load_iris

dataset = load_iris()       #dataset di prova che ci fornisce sklearn

X = dataset.data
y = dataset.target

print(X.shape)
print(y.shape)

#adesso utilizzo 4 variabili dove inserire i rispettivi valori di training e di test dalla funzione
X_train, X_test, y_train, y_test = train_test_split(X, y) #questo metodo restituirà automaticamente 4 valori

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


#Per default si avrà il 75% assegnato al training e il 25% assegnato al test
#volendo si può decidere di modificare tale opzione

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=40)
#cosi sto dicendo che voglio 40 valori nella parte di training

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)