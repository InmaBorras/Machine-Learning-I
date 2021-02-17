#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:19:03 2021

@author: inma
"""
# Logistic regresion

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


#requires feature data to be normalized

# Create logistic regression model
model = LogisticRegression()

#Train the model
model.fit(features, labels)#The first is a matrix of features, and the second is a matrix of class labels. 

#predicion of the class
model.predict(features)#returns a vector of labels, sklearn uses a classification threshold of 0.5 at standard

#prediction of the probability 

model.predict_proba(features)# ranging from 0 to 1, for each sample

#Calcular los coeficientes para ver la influencia de las features en el modelo
coefficients=model.coef_
intercept= model.intercept_

#Visualizar los coeficientes como una lista
coefficients=coefficients.tolist()[0]


#representacion de los coeficientes cuanto mas positivo o mas negativo mas influencia.
plt.bar([1,2],coefficients)
plt.xticks([1,2],['feature','feature'])#poner aqui las featurres que queremos ver 
plt.xlabel('feature')
plt.ylabel('coefficient')
 
plt.show()




# KNN 

from sklearn.neighbors import KNeighborsClassifier



classifier = KNeighborsClassifier(n_neighbors = 3)# n_neighbors is K

#Train the classifier
classifier.fit(training_points, training_labels)

#aplicar el modelo con los puntos desconocidos
guesses = classifier.predict(unknown_points)# se peuden poner los que queramos

#calculo de la mejor K 
def find_validation_accuracy(training_set, training_labels, validation_set, validation_labels, k):
  num_correct = 0.0
  for title in validation_set:
    guess = classify(validation_set[title], training_set, training_labels, k)# creo que se debe sustituir por la variable guesses
    if guess == validation_labels[title]:#aqui igual sustituirpr la variable guesses 
      num_correct += 1
  return num_correct / len(validation_set)


# creo que se pueder representar todas las K para elegir la mejor 


#Decision Trees 







