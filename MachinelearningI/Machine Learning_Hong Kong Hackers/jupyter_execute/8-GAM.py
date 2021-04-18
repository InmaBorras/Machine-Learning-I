#!/usr/bin/env python
# coding: utf-8

# # 7. Modelos Aditivos Generalizados (GAM)
# 
# 
# 
# Utilizamos un modelo de clasificacion de GAM  para identiciar si las casas son caras o baratas como habiamos visto anteriormente en los demás modelos.
# 
# El modelo de clasificación de GAM se rige por la siguiente ecuación. 

# ![image.png](attachment:image.png)

# Primero  dividiremos los datos en train y test, para entrerar y probar los diferentes ajustes del modelo y finalmente realizaremos una validación del modelo elegido con los datos de validación. 
# 
# 

# In[1]:


import numpy as np
import pandas as pd
from patsy import dmatrix
from matplotlib import pyplot as plt

import statsmodels.api as sm
from pygam import LinearGAM, s, GAM, l, te


# In[2]:


# cargamos los datos
data_RF = pd.read_csv("./CSV/csv_precio_div.csv",sep=',')

data_RF = data_RF.drop('Unnamed: 0',axis=1)#eliminamos la primera columna. 

columns=['Distancia_NEW','Landsize','Longtitude','Lattitude','precio_div']
data=(data_RF[columns])


# Hemos separado  el 70%

# dividir el data set de forma aleatoria 

p_train = 0.70 # Porcentaje de train.

data['is_train'] = np.random.uniform(0, 1, len(data)) <= p_train
train, test = data[data['is_train']==True], data[data['is_train']==False]
df = data.drop('is_train', 1)

print("Ejemplos usados para entrenar: ", len(train))
print("Ejemplos usados para test: ", len(test))


# In[3]:


#Seleccion de varaibles
X = train[['Distancia_NEW','Landsize','Lattitude','Longtitude']].values
y = train['precio_div'].values

X_test=test[['Distancia_NEW','Landsize','Lattitude','Longtitude']].values
y_test=test['precio_div'].values

y =np.array(y).reshape(-1, 1)
y_test=y_test.reshape(-1, 1)


# In[4]:


#NORMALIZACION
#from sklearn import preprocessing
#scaler = preprocessing.StandardScaler().fit(X)
#X= scaler.transform(X)
#X_test = scaler.transform(X_test)


# ## 7.1  Ajuste de hiperparámetros
# 
# En primerlugar, incorporamos todas las variables como splines en el modelo. 

# In[5]:


gam = LogisticGAM(s(0)+ s(1)+s(2)+ s(3)).gridsearch(X, y)


gam.summary()


print("----------------------------------------------------------------------------------------")
print("GRAFICAS DE DEPENDENCIA PARCIAL")

fig, axs = plt.subplots(1, 4, figsize=(15,6))
titles = ['Distancia_NEW','Landsize','Lattitude','Longtitude']

for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, width=.95)

    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], confi, c='r', ls='--')
    ax.set_title(titles[i]);



# In[155]:



#PRECISION DEL MODELO

pred= gam.predict(X_test)
print(classification_report(y_test, pred))
print (" Accuracy del modelo con los datos de Train",gam.accuracy(X, y))
print (" Accuracy del modelo con los datos de Test", gam.accuracy(X_test, y_test))


# En las gráficas de dependencia parcial podemos ver como cada una de las variables afecta al precio.La variable ' distance_NEW' tiene una correlación negativa con el precio, por lo tanto, cuanto mayor es esta más bajo será el precio. 
# 
# Con repecto a ' Latitude' y ' Longitude' podemos observar que  el precio varia en fución del punto en el que se encuentre, mientra que el 'Landsize' podemos considerar que la correlación es positiva, ya que cuanto mayor es el tamano de landsize mas caro es el precio. 
# 
#  Aplicamos tensor a Longitud y latitud ya que son dos variables que  están muy relacionadas. 
#  

# In[156]:


#Aplicamos tensor
gam_te = LogisticGAM(s(0)+ s(1)+te(2,3)).gridsearch(X, y)

gam_te.summary()


# In[133]:


fig, axs = plt.subplots(1, 2, figsize=(15,6))
titles = ['Distancia_NEW','Landsize']

for i, ax in enumerate(axs):
    XX = gam_te.generate_X_grid(term=i)
    pdep, confi = gam_te.partial_dependence(term=i, width=.95)

    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], confi, c='r', ls='--')
    ax.set_title(titles[i]);


# In[139]:


# Solo el tensor 
gam_te = LogisticGAM(s(0)+ te(2,3)+s(1)).gridsearch(X, y)
XX = gam_te.generate_X_grid(term=1, meshgrid=True)
Z = gam_te.partial_dependence(term=1, X=XX, meshgrid=True)

ax = plt.axes(projection='3d')
ax.plot_surface(XX[0], XX[1], Z, cmap='viridis')


# Podemos observar que hay ciertas combinaciones de latitudes y longitudes donde el precio  tiende a ser menor en el centro de la grafica, sin embargo, otras presentan un  una relacion más positiva con el precio en los extremos. 

# In[157]:


#PRECISION DEL MODELO

pred= gam_te.predict(X_test)
print(classification_report(y_test, pred))
print (" Accuracy del modelo con los datos de Train", gam_te.accuracy(X, y))
print (" Accuracy del modelo con los datos de Test", gam_te.accuracy(X_test, y_test))


# Con el uso del tensor en las varibles 'lattitude' y ' longtitude' , aumenta  el valor de accuracy tanto en  en en los datos de train como en los de test. El valor f1-score se encuentra equilibrado entre los entre las dos clases. 
# 
# 
# Después de comprobar que el uso de un tensor es beneficioso para  el ajuste del modelo, añadiremos una by-varaible sobre "latitudde" la variable "longitude".

# In[158]:


#Interacción simple con by 
gam_by = LogisticGAM(s(0)+ s(1)+s(3, by=2)).gridsearch(X, y)# interaccions simple

gam_by.summary()
print('-------------------------------------------------------------------------------------')
print (" Accuracy del modelo con los datos de Train", gam_by.accuracy(X, y))


# In[160]:


#PRECISION DEL MODELO

pred= gam_by.predict(X_test)
print(classification_report(y_test, pred))
print (" Accuracy del modelo con los datos de Train", gam_by.accuracy(X, y))
print (" Accuracy del modelo con los datos de Test", gam_by.accuracy(X_test, y_test))


# Aunque el resultado es mejor que el de las dos variables por separado, tanto en los datos de train como los de test tiene un mejor ajuste del  modelo en cuando usamos el tensor. Seguiremos ajustando el modelo haciendo uso del tensor en  las variables anteriormente mencionadas. 
# 
# 
# ### Ajuste de hiperparámetros 
# 
# Ajustamos los hiperparámetros para ver como afecta el número de splines(n_splines), si es necesario  necesario añadir una costante(fit_intercept), el número de interacciones permitidas (max_inter)y  lambda (lam). 
# 
# #### Lambda 
# 
# 
# Seleccionamos los mejores lambdas de entre un rango aleatorio priemro y después probamos con 0.6 ya que suele ser el valor estandar. 

# In[161]:


#MODELO 1 

#Ajustamos Lambda aleatorio

lams = np.random.rand(100, 4)
lams = lams * 11 - 3
lams = np.exp(lams)
#print(lams.shape)

fit_intercept=True
n_splines=10
max_iter=100

model = LogisticGAM(s(0)+ s(1)+te(2,3), fit_intercept=fit_intercept , n_splines=n_splines, max_iter=max_iter).gridsearch(X, y,  lam=lams)
model.summary()
print("_________________________________________________________________________________________________")
print (" Accuracy del modelo con los datos de Train del Modelo 1 ", model.accuracy(X_test, y_test))


# In[162]:


#PRECISION DEL MODELO

pred= model.predict(X_test)
print(classification_report(y_test, pred))
print (" Accuracy del modelo con los datos de Train", model.accuracy(X, y))
print (" Accuracy del modelo con los datos de Test", model.accuracy(X_test, y_test))


# In[163]:


#MODELO 1 

#Ajustamos Lambda valor fijo 0.6

lams=[0.6,0.6,0.6,0.6]

fit_intercept=True
n_splines=10
max_iter=100

model = LogisticGAM(s(0)+ s(1)+te(2,3), fit_intercept=fit_intercept , n_splines=n_splines, max_iter=max_iter).gridsearch(X, y,  lam=lams)
model.summary()
print("_________________________________________________________________________________________________")
print (" Accuracy del modelo con los datos de Train del Modelo 1 ", model.accuracy(X_test, y_test))


# In[164]:


#PRECISION DEL MODELO

pred= model.predict(X_test)
print(classification_report(y_test, pred))
print (" Accuracy del modelo con los datos de Train", model.accuracy(X, y))
print (" Accuracy del modelo con los datos de Test", model.accuracy(X_test, y_test))


# La exactitud de ambos modelos son  muy parecidos pero  en ambos casos es peor que el modelo inicial usando  el rango de lambdas aleatorios, aplicaremos este  metodo en los siguientes ajuste del modelo. 
# 
# #### Ajuste del parámetro fit_intercept 
# 
# 
# Probamos con los valores de False y True. 

# In[166]:


#MODELO 2 
#Ajustamos Lambda aleatorio

lams = np.random.rand(100, 4)
lams = lams * 11 - 3
lams = np.exp(lams)

fit_intercept= False 
n_splines=10
max_iter=100

model_2 = LogisticGAM(s(0)+ s(1)+te(2,3), fit_intercept=fit_intercept , n_splines=n_splines, max_iter=max_iter).gridsearch(X, y,  lam=lams)
model_2.summary()
print("_________________________________________________________________________________________________")
print (" Accuracy del modelo con los datos de Train del Modelo 2 ", model_2.accuracy(X_test, y_test))


# In[167]:


#PRECISION DEL MODELO

pred= model_2.predict(X_test)
print(classification_report(y_test, pred))
print (" Accuracy del modelo con los datos de Train", model_2.accuracy(X, y))
print (" Accuracy del modelo con los datos de Test", model_2.accuracy(X_test, y_test))


# Vemos que en este caso el uso del intercep o no, no influye en mucho , pero el modelo con intercep parece que es ligeramente mejor. 
# 
# #### Ajuste del parámetro n_splines 
# 
# Modificamos el número de splines para observar si mejora el resultado sin llegar al sobreajuste. 

# In[178]:


#MODELO 3

lams = np.random.rand(100, 4)
lams = lams * 11 - 3
lams = np.exp(lams)

fit_intercept= True 
n_splines=15
max_iter=100

model_3 = LogisticGAM(s(0)+ s(1)+te(2,3), fit_intercept=fit_intercept , n_splines=n_splines, max_iter=max_iter).gridsearch(X, y,  lam=lams)
model_3.summary()
print("_________________________________________________________________________________________________")
print (" Accuracy del modelo con los datos de Train del Modelo 3 ", model_3.accuracy(X_test, y_test))


# In[182]:


pred=model_3.predict(X_test)
print(classification_report( y_test, pred))
print( confusion_matrix )
print (" Accuracy del modelo con los datos de Train del Modelo 3 ", model_3.accuracy(X, y))
print (" Accuracy del modelo con los datos de Test del Modelo 3 ", model_3.accuracy(X_test, y_test))


# In[180]:


fig, axs = plt.subplots(1, 2, figsize=(15,6))
titles = ['Distancia_NEW','Landsize']

for i, ax in enumerate(axs):
    XX = model_3.generate_X_grid(term=i)
    pdep, confi = model_3.partial_dependence(term=i, width=.95)

    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], confi, c='r', ls='--')
    ax.set_title(titles[i]);


# In[176]:


# Solo el tensor 
lams = np.random.rand(100, 4)
lams = lams * 11 - 3
lams = np.exp(lams)

fit_intercept= True 
n_splines=15
max_iter=100

model_3 = LogisticGAM(s(0)+te(2,3)+ s(1), fit_intercept=fit_intercept , n_splines=n_splines, max_iter=max_iter).gridsearch(X, y,  lam=lams)

XX = model_3.generate_X_grid(term=1, meshgrid=True)
Z = model_3.partial_dependence(term=1, X=XX, meshgrid=True)

ax = plt.axes(projection='3d')
ax.plot_surface(XX[0], XX[1], Z, cmap='viridis')


# Al aumentar el numero de splines, también conseguimos un mejor rendimiento del modelo. Sin embargo el tiempo de ejecución aumenta mucho. 
# 
# Aumetaremos un poco más el número de splines para incrementar un poco mas la exactitud  pero sin peligro de sobreajuste.

# In[185]:


#MODELO 4
lams = np.random.rand(100, 4)
lams = lams * 11 - 3
lams = np.exp(lams)

fit_intercept= True 
n_splines=20
max_iter=100

model_4 = LogisticGAM(s(0)+ s(1)+te(2,3), fit_intercept=fit_intercept , n_splines=n_splines, max_iter=max_iter).gridsearch(X, y,  lam=lams)
model_4.summary()
print("_________________________________________________________________________________________________")
print (" Accuracy del modelo con los datos de Train del Modelo 4 ", model_4.accuracy(X_train, y_train))


# In[201]:


pred=model_4.predict(X_test)
print(classification_report( y_test, pred))
print( confusion_matrix )
print (" Accuracy del modelo con los datos de Train del Modelo 4 ", model_4.accuracy(X, y))
print (" Accuracy del modelo con los datos de Test del Modelo 4 ", model_4.accuracy(X_test, y_test))


# Finalmente obsevamos que en el Modelo 4 obtenemos merores resutados en el ajuste del modelo, procederemos a pintar las gráficas de dependencia para comprobar la interpretabilidad del modelo. 

# In[188]:


fig, axs = plt.subplots(1, 2, figsize=(15,6))
titles = ['Distancia_NEW','Landsize']

for i, ax in enumerate(axs):
    XX = model_4.generate_X_grid(term=i)
    pdep, confi = model_4.partial_dependence(term=i, width=.95)

    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], confi, c='r', ls='--')
    ax.set_title(titles[i]);


# In[189]:


model_4 = LogisticGAM(s(0)+te(2,3)+ s(1)).gridsearch(X, y)
XX = model_4.generate_X_grid(term=1, meshgrid=True)
Z = model_4.partial_dependence(term=1, X=XX, meshgrid=True)

ax = plt.axes(projection='3d')
ax.plot_surface(XX[0], XX[1], Z, cmap='viridis')


# Las gráficas de dependencia parecen igualmente interpretables como visualizamos en le primero de los modelos con la misma correlacion. En la gráfica del tensor,  podemos ver en dferencia la anterior ademas de la zona centrica de la gráfica con los precios bajo , tambien  hay una zona delimitada con precios más caros, que seguramente corresponde a la zona cara de la ciudad. 
# 
# 
# ## 7.2. Validación del modelo entrenado 
# 
# 
# Una vez seleccionado el modelo  de GAM más optimo procedemos ha aplicar el modelo entrenado en los datos de validación para comprobar como se comporta. 

# In[206]:


data = pd.read_csv('./CSV/csv_precio_div_validation.csv')
columns=['Distancia_NEW','Landsize','Longtitude','Lattitude','precio_div']

Validation=(data[columns])


#Seleccion de varaibles
X_val = Validation[['Distancia_NEW','Landsize','Lattitude','Longtitude']].values
y_val = Validation['precio_div'].values


# In[207]:


pred=model_4.predict(X_val)
print(classification_report( y_val, pred))
print( confusion_matrix )
print (" Accuracy del modelo con los datos de Train del Modelo 4 ", model_4.accuracy(X, y))
print (" Accuracy del modelo con los datos de Test del Modelo 4 ", model_4.accuracy(X_val, y_val))


# Obtenemos un buen resultado  en la la exactitud de la prediccion de los datos de validación, por lo que pododemos concluir que es el  Modelo 4 con el ajuste de hiperparámetros previamente hecho es el que mejor se ajusta a estos datos entre los que se han probado. 

# In[ ]:




