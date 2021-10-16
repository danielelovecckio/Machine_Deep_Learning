# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 17:30:18 2020

@author: daniele
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

dataset = pd.read_csv('prezzi_case_class.csv')

print(dataset)
