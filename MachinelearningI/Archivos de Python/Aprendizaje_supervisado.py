#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:19:03 2021

@author: inma
"""
### LIMPIEZA DE DATOS 
import numpy as np
import pandas as pd

###VISUALIZACION DEL MODELO

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sb
 
#%matplotlib inline
# plt.rcParams['figure.figsize'] = (16, 9)
# plt.style.use('ggplot')


###REGRESION lOGISTICA

from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

### KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn import preprocessing


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


###SVM 
from sklearn.svm import SVC
from sklearn import svm
from sklearn.naive_bayes import GaussianNB


###DIVISION DE LOS DATOS----------------------------------------------- 

data= pd.read_csv("/home/inma/Master_Data_Science _Contenido/Machine_learningI/Practica/Machine-Learning-I/CSV/Seleccion_variables_RF.csv")


#Etiquetarlos datos en funcion de alto o bajo. 

mediana= 870000.0
data["precio_div"]= np.where(data["Price"]>=mediana, "1", "0" )#1 es caro Y 0 es barato 
#print(data.head(100))
data=data.replace(np.nan,"0")

# ver cuantos tenemos de cada clase 

#print(data.groupby('precio_div').size())

# dividir el data set de forma aleatoria 

p_train = 0.70 # Porcentaje de train.

data['is_train'] = np.random.uniform(0, 1, len(data)) <= p_train
train, test = data[data['is_train']==True], data[data['is_train']==False]
df = data.drop('is_train', 1)

#print(train.head())

#print("Ejemplos usados para entrenar: ", len(train))
#print("Ejemplos usados para test: ", len(test))

# #NORMALIZACION

# X = train[['Distance','Lattitude','Landsize','Bathroom','precio_div']]#.values
# print(X)
# X= pd.DataFrame(X[['Distance','Lattitude','Landsize','Bathroom']])
# print(X.head())



# scaler = MinMaxScaler()
# train= scaler.fit_transform(train)
# test = scaler.transform(test)

# datos= pd.DataFrame(datos)
# print(datos.head())

 # SELECCION DE VARIABLES

# X = train[['Distance','Lattitude','Landsize','Bathroom']].values
# y = train[['precio_div']].values

# X_test=test[['Distance','Lattitude','Landsize','Bathroom']].values
# y_test=test[['precio_div']]
 


# ####REPRESENTACION DE LOS DATOS 

# sb.set(style="whitegrid", palette="husl")
# #data = sb.load_dataset("data")

# # Long format to plot
# datos = pd.melt(datos, "precio_div", var_name="Variables")#a veces hay que pasasr de wide a long para poder  ver losa datos  y que lo haga solo
# print("hola")
# f, ax = plt.subplots(1, figsize=(15,10))
# sb.stripplot(x="Variables", y="value ", hue="precio_div", data=datos, jitter=True, edgecolor="white", ax=ax);

###REGRESION LOGISTICA -----------------------------------------------------------------


# #Train 
# # crear los features y los labels 

# features = np.array(train[['Distance','Lattitude','Landsize','Bathroom']])
# labels = np.array(train[['precio_div']])

# #test
# #crear los features y los labels para la validacion

# features_t = np.array(test[['Distance','Lattitude','Landsize','Bathroom']])
# labels_t = np.array(test[['precio_div']])

# labels.shape


# # Create logistic regression model

# model = linear_model.LogisticRegression()


# #Train the model
# model.fit(features, labels)#The first is a matrix of features, and the second is a matrix of class labels. 


# '''
# # #predicion of the class usando los mismos datos 
# # predictions= model.predict(features)#returns a vector of labels, sklearn uses a classification threshold of 0.5 at standard
# # print(predictions)

# # # Precision media de las predicciones

# # model.score(features,labels)
# '''

# #Validacion del modelo

# #predicion of the class
# predictions_t= model.predict(features_t)#returns a vector of labels, sklearn uses a classification threshold of 0.5 at standard
# #print(accuracy_score(labels_t, predictions_t))


# #Reporte de resultados del Modelo

# #print(classification_report(labels_t , predictions_t))




'''    
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

porcentaje de acierto del modelo
model.score(feature,lables)

'''





### KNN------------------------------------------------ 


# # SELECCION DE VARIABLES

# X = train[['Distance','Lattitude','Landsize','Bathroom']].values
# y = train[['precio_div']].values

# X_test=test[['Distance','Lattitude','Landsize','Bathroom']].values
# y_test=test[['precio_div']]
 
# #NORMALIZACION


# scaler = MinMaxScaler()
# X= scaler.fit_transform(X)
# X_test = scaler.transform(X_test)



# '''

# # scaler= preprocessing.Normalizer(norm='l2', copy=True)
# # train[['Distance','Lattitude','Landsize','Bathroom']]=scaler.fit_transform(train[['Distance','Lattitude','Landsize','Bathroom']])
# # test[['Distance','Lattitude','Landsize','Bathroom']]=scaler.fit_transform(test[['Distance','Lattitude','Landsize','Bathroom']])


# #
# # print("Minimo", train[['Distance','Lattitude','Landsize','Bathroom']].min())


# # features=train[['Distance','Lattitude','Landsize','Bathroom']]
# # features_t=test[['Distance','Lattitude','Landsize','Bathroom']]
# # labels = np.array(train[['precio_div']])
# # labels_t = np.array(test[['precio_div']])
# '''


# #ENTRENAR EL MODELO

# n_neighbors = 3
# classifier = KNeighborsClassifier(n_neighbors)# n_neighbors is K

# #Train the classifier
# classifier.fit(X,y)
# print('Accuracy of K-NN classifier on training set: {:.2f}'.format(classifier.score(X, y)))
# print('Accuracy of K-NN classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

# #PRECISION DEL MODELO

# pred = classifier.predict(X_test)
# print(confusion_matrix(y_test, pred))
# print(classification_report(y_test, pred))

# VISUALIZACION DEL MODELO 


# h = .02  # step size in the mesh
# # Create color maps
# cmap_light = ListedColormap(['#FFAAAA', '#ffcc99', '#ffffb3','#b3ffff','#c2f0c2'])
# cmap_bold = ListedColormap(['#FF0000', '#ff9933','#FFFF00','#00ffff','#00FF00'])
 
# # we create an instance of Neighbours Classifier and fit the data.
# clf = KNeighborsClassifier(n_neighbors, weights='distance')
# clf.fit(X, y)
 
# # Plot the decision boundary. For that, we will assign a color to each
# # point in the mesh [x_min, x_max]x[y_min, y_max].
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
 
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure()
# plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
 
# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
#                 edgecolor='k', s=20)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
    
# patch0 = mpatches.Patch(color='#FF0000', label='1')
# patch1 = mpatches.Patch(color='#ff9933', label='2')
# patch2 = mpatches.Patch(color='#FFFF00', label='3')
# patch3 = mpatches.Patch(color='#00ffff', label='4')
# patch4 = mpatches.Patch(color='#00FF00', label='5')
# plt.legend(handles=[patch0, patch1, patch2, patch3,patch4])
 
    
# plt.title("5-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))
 
# plt.show()


# #ELEGIR EL MEJOR K 

# k_range = range(1, 20)
# scores = []
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors = k)
#     knn.fit(X, y)
#     scores.append(knn.score(X_test, y_test))
# plt.figure()
# plt.xlabel('k')
# plt.ylabel('accuracy')
# plt.scatter(k_range, scores)
# plt.xticks([0,5,10,15,20])

# print(plt)# como se imprime esto solo??



#SVM --------------------------------------------------------------


#Train 
# crear los features y los labels 

# X = np.array(train[['Distance','Lattitude','Landsize','Bathroom']])
# Y = np.array(train[['precio_div']])

# #test
# #crear los features y los labels para la validacion

# X_test = np.array(test[['Distance','Lattitude','Landsize','Bathroom']])
# Y_test = np.array(test[['precio_div']])

# #Kernel no lineal
# clf=SVC()
# clf.fit(X,Y)
# len(Y_test)
# #737
# P=clf.predict(X_test)
# print(sum(P==Y_test)/(len(Y_test)))

# #Kernel Lineal
# lin_clf=svm.LinearSVC()
# lin_clf.fit(X,Y)
# Pred=lin_clf.predict(X_test)
# print(sum(Pred==Y_test)/(len(Y_test)))
# #0.9701




X = train[['Distance','Lattitude','Landsize','Bathroom']].values
y = train['precio_div'].values

X_test=test[['Distance','Lattitude','Landsize','Bathroom']].values
y_test=test['precio_div'].values

# classifier = SVC(kernel = "rbf", gamma = 0.3, C = 2)
# classifier.fit(X, y) 
# #print(classifier.predict(X_test))
# print("Kernel rbf", classifier.score(X_test, y_test))




# classifier = SVC(kernel = "linear", C = 2)
# classifier.fit(X, y) 
# #print(classifier.predict(X_test))
# print("Kernel lineal", classifier.score(X_test, y_test))

from sklearn.model_selection import GridSearchCV

#define the hyperparameters we ant to tune 
param_grid={
            'kernel':['rbf'],
            'C':[0.001,0.01,0.1,1,10],
            'gamma': [0.001,0.01,0.1,1],
}

#instantiate GridSearchCV fit model, and male prediction

gs_svc=GridSearchCV(SVC(), param_grid=param_grid)
gs_svc.fit(X,y)
y_pred=gs_svc.predict(X_test)







#CODE ACADEMY 


# #Kernel Lineal
# classifier = SVC(kernel = 'linear', c=0.01)
# #classifier = SVC(kernel = "rbf", gamma = 0.3, C = 2)
# #classifier.fit(X, Y) 
# print(classifier.predict(X_test))
# print(classifier.score(X_test, Y_test))


# classifier = SVC(kernel = "rbf", gamma = 0.5, C = 2)
# classifier = SVC(kernel = "poly", degree = 2)