#!/usr/bin/env python
# coding: utf-8

# # 3. Aprendizaje Supervisado 
# 
# 
# Realizaremos disintintos modelos de aprendizaje supervisado, ajustnado sus hiperparametros para posteriormente ver cual de ellos  presenta un mejor desmpeño. 
# 
# 

# In[1]:


### LIMPIEZA DE DATOS 
import numpy as np
import pandas as pd

###VISUALIZACION DEL MODELO

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sb


# ## Modelos de Aprendizaje Supervisado

# Creamos una nueva variable que se  categorizará el precio en funcion de si es caro(1) o barato (0). 

# In[2]:


###DIVISION DE LOS DATOS----------------------------------------------- 

data= pd.read_csv("seleccion_variables_RF_bathandrooms.csv")


#Etiquetarlos datos en funcion de alto o bajo. 

mediana= 870000.0
data["precio_div"]= np.where(data["Price"]>=mediana, "1", "0" )#1 es caro Y 0 es barato 
#print(data.head(100))
#data=data.replace(np.nan,"0")

data.to_csv('csv_precio_div')


print(data.groupby('precio_div').size())

Preparamos los dataset de training y de test. 
# In[3]:


# Hemos separado  el 70%

# dividir el data set de forma aleatoria 

p_train = 0.70 # Porcentaje de train.

data['is_train'] = np.random.uniform(0, 1, len(data)) <= p_train
train, test = data[data['is_train']==True], data[data['is_train']==False]
df = data.drop('is_train', 1)

print("Ejemplos usados para entrenar: ", len(train))
print("Ejemplos usados para test: ", len(test))


# ### 1. GLM : Regresion Logística 
# 
# 
# 

# Seleccionamos las variables  seleccionada a raiz del PCA 'Distancia_NEW','Lattitude','Longtitude','Landsize' e intentamos predecir la variable categorizada del precio. 

# In[4]:


from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[5]:


#TRAIN 
features = np.array(train[['Distancia_NEW','Lattitude','Longtitude','Landsize']])
labels = np.array(train['precio_div'])


# In[6]:


#TEST
features_t = np.array(test[['Distancia_NEW','Lattitude','Longtitude','Landsize']])
labels_t = np.array(test['precio_div'])


# In[7]:


# Create logistic regression model

model = linear_model.LogisticRegression()

#Train the model
model.fit(features, labels)#The first is a matrix of features, and the second is a matrix of class labels. 


# Evaluamos como de bueno ha sido el ajuste del modelo sobre los propios datos de entrenamiento.  

# In[8]:


predictions= model.predict(features)
print(accuracy_score(labels, predictions))


# Validamos el modelos usando los datos de test. 

# In[9]:


#Validacion del modelo
predictions_t= model.predict(features_t)
print(accuracy_score(labels_t, predictions_t))


# Para obtener mas información sobre el modelo  procedemos a evaluar los resultados con un matriz de confusión  y un reporte de clasificación.  

# #### 1.1. Evaluación del modelo de Regresión Logística

# In[10]:


#Reporte de resultados del Modelo

print(classification_report(labels_t , predictions_t))


# Podemos observar que la distribución entre ambas categorias es bastante homogénea,  al igual que el acierto en la predicción de cada una de ellas es parecida aunque ligeramente superior en la categoría de casas caras. 
# 
# F1 score  es 0.62  un valor que  se puede considerar como aceptable. 

# In[11]:


#dibujo de la Curva ROC

from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import plot_roc_curve
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

rlog_disp = plot_roc_curve(model, features, labels)
plt.show()


# ### 2. K- NEAREST NEIGHBORS (KNN )
# 
# 

# In[12]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[13]:


# # SELECCION DE VARIABLES

X = train[['Distancia_NEW','Lattitude','Longtitude','Landsize']].values
y = train['precio_div'].values

X_test=test[['Distancia_NEW','Lattitude','Longtitude','Landsize']].values
y_test=test['precio_div'].values
 


# Para realizar  el modelo de Knn primero estandarizamos los datos. 

# In[14]:


#NORMALIZACION
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)


# #### 2.1. Ajuste de hiperparametros 

# Con el objetivo de elegir el mejor número de vecinos dibujamos una gráfica con todas las variables en un rango entre 0 y 20. 

# In[15]:


#ELEGIR EL MEJOR K 

k_range = range(1, 20)
scores = []
for k in k_range:
     knn = KNeighborsClassifier(n_neighbors = k)
     knn.fit(X, y)
     scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])


# Visualizando la gráfica anterior buscamos el valor de k que nos proporcione el mejor f1-score dentro del rango entre 1 y 10 vecinos. 

# In[16]:



#encontrar la mejor k 

best_k=0
best_score=0
neighbors=range(1,10,2)#considerara min_k=1, max_k=10, solo odd numbers 
for k in neighbors:
    knn=KNeighborsClassifier(n_neighbors=k) #instantiante classifier
    knn.fit(X, y )# fit model 
    knn_y_pred= knn.predict(X_test)
    
    #we will consider the optimal k to be the k that produce the highest f1 score
    f1 = metrics.f1_score(y_test, knn_y_pred, pos_label='1')
    if f1> best_score:
        best_k= k 
        best_score =f1
        
print (best_k)


# Obtenemos que el mejor valor para k y por lo tanto procedemos a  entrenar nuestro modelo con el. Este dato corresponde con la gráfica anteriormente  visualizada. 

# In[17]:


#instantiate the classifier with the optimal k, fir the model and make predictions
knn=  KNeighborsClassifier(n_neighbors=best_k) 
knn.fit(X, y )
knn_y_pred= knn.predict(X_test)


# #### 2.2 Evaluación del modelo de KNN

# In[18]:


#Hiperparametros-
n_neighbors = best_k
#algorithm='brute'
#p=1
weights='distance'# uniforme ( todos los puntos son iguales )
#n_jobs=-1

classifier = KNeighborsClassifier(n_neighbors)

# #Train the classifier
classifier.fit(X,y)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(classifier.score(X, y)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))


# In[19]:


#PRECISION DEL MODELO
pred = classifier.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# Obtenemos un f1-score de 0.81 lo cual es bastante bueno. 

# Representamos la curva ROC para ver como se comporta el modelo en relación a falsos positivos y  verdaderos positivos. 

# In[20]:


#Pintar curva ROC
Knn_disp = plot_roc_curve(classifier, X, y)
plt.show()


# ### 3. SVM 
# 
# 
# 
# 
# Primero sin normalizar y despues normalizando 
# probar los difrentes kernels 

# In[21]:


from sklearn.svm import SVC
from sklearn import svm
from sklearn.naive_bayes import GaussianNB


# In[22]:


X = train[['Distancia_NEW','Lattitude','Longtitude','Landsize']].values
y = train['precio_div'].values

X_test=test[['Distancia_NEW','Lattitude','Longtitude','Landsize']].values
y_test=test['precio_div'].values


# Realizamos el modelo de SVM con los tres tipos de Kernel  más comunes Lineal, polinomico y rbf. Los tres con los mismos valores de C para evaluar cual tiene mejor resultado. 

# Kernel Lineal 

# In[23]:


#classifier = SVC(kernel = "linear", C = 2)
#classifier.fit(X, y) 
#print(classifier.predict(X_test))
#print("Kernel lineal", classifier.score(X_test, Y_test))


# Kernel Polinómico

# In[24]:


#classifier = SVC(kernel = "poly",degree= 2, C = 2)
#classifier.fit(X, y) 
#print(classifier.predict(X_test))
#print("Kernel polinómico", classifier.score(X_test, Y_test))


# Kernel RBF

# In[25]:


classifier = SVC(kernel = "rbf", gamma = 0.3, C = 2)
classifier.fit(X, y) 
#print(classifier.predict(X_test))
print("Kernel rbf", classifier.score(X_test, y_test))


# El modelo que mejor resultado obtiene es el propuesto con kernel "rbf". Normalizamos los datos para observar si mejora el rendimiento.

# In[24]:


#NORMALIZACION

scaler = preprocessing.StandardScaler().fit(X)
X_norm = scaler.transform(X)
X_test_norm = scaler.transform(X_test)


#Prediccion con datos normalizados 


classifier = SVC(kernel = "rbf", gamma = 0.3, C = 2)
classifier.fit(X_norm, y) 
#print(classifier.predict(X_test))
print(" Kernel rbf con datos normalizados: ", classifier.score(X_test_norm, y_test))


# #### 3.1.Ajuste de hiperparametros
# 
# Hemos obtenido que normalizando los datos y usando el Kernel 'rbf'  obtenemos los mejores resultados , partiremos de esa base para el cálculo de hiperparámetros.
# 
# Aplicamos esta técnica para probar diferentes C y Gammas. 

# In[25]:


from sklearn.model_selection import GridSearchCV

#define the hyperparameters we want to tune 
param_grid={
            'kernel':['rbf'],
            'C':[0.001,0.01,0.1,1,10],
            'gamma': [0.001,0.01,0.1,1],
            }

#instantiate GridSearchCV fit model, and male prediction

gs_svc=GridSearchCV(SVC(), param_grid=param_grid)
gs_svc.fit(X_norm,y)
c=gs_svc.predict(X_test_norm)


# In[26]:


print(param_grid)


# In[31]:


print(gs_svc.score(X_test_norm, y_test))


# Tras el ajuste de los hiperparámetros el resultado obtenido por svm  alcanza mi un porcetaje muy elevado. 

# In[35]:


#PRECISION DEL MODELO
pred = gs_svc.predict(X_test_norm)
print(classification_report(y_test, pred))


# In[30]:


svm_roc = plot_roc_curve(gs_svc, X_norm, y)
plt.show()


# ## Evaluacion del punto de corte 
# 
# Para elegir el mejor punto de corte  para la categorizacion binaria del precio de las casas en caras y baratas vamos a utilzar la curva ROC. 
# 
# 
# En el siguiente gráfico, Figura 1, se representa la curva ROC y el punto de
# corte que maximiza el K-S, que se corresponde con el punto en la curva
# ROC cuya distancia horizontal al eje es máxima 

# In[36]:


from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import plot_roc_curve
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

ax = plt.gca()
rlog_disp.plot(ax=ax, alpha=0.8)
Knn_disp.plot(ax=ax, alpha=0.8)
svm_roc.plot(ax=ax, alpha=0.8)
plt.show()


# In[ ]:




