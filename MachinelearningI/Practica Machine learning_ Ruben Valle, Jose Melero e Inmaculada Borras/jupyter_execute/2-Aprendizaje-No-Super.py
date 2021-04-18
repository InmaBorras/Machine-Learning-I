#!/usr/bin/env python
# coding: utf-8

# # 2. Aprendizaje no supervisado 
# 
# 
# ## 2.1.Objetivo
# 
# El objetivo de este apartado es comprobar como se hace una mejor "Clusterización" de los pisos usando KMeans. Intentaremos deducir si la mediana que (está en 875000 $) es un valor adecuado para catalogar un piso en caro o barato. Para ellos vamos a comparar los resultados de clusterizar usando Kmeans, empleando como entrada  tres variables obtendias mediante TSNE sobre el dataframente original. Después llevaremos a cabo lo mismo pero empleando las tres variables con más peso en relación al precio que se vieron en la práctica de de regresión lineal y como último paso añadiremos una cuarta variable, ya que tras algunas pruebas ha demostrado hacer una buena clasificacion de los pisos.
# 
# Cargamos las librerias que vamos a necesitar  y el archivo donde se encuetran las funiones que usaremos posteriormente. 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sb
import warnings
warnings.filterwarnings('ignore')
import os
os.getcwd()
from scipy import stats
def close_to_far_from_center(X,centroid, n=10):
    
    distance = np.sum((X - centroid) ** 2, axis=1)
    
    print('Close to center')
    display(data_RF.iloc[np.argsort(distance)[:n]])
    
    print('Far from center')
    display(data_RF.iloc[np.argsort(distance)[-n:]])

from src.utils import pd, np, plt
import  src.utils 

# Useful functions
from src.utils import load_examples, plot_scatter, plot_silhouette



# Cargamos los datos con las variables del dataframe.

# In[2]:


# cargamos los datos

data_RF = pd.read_csv("./CSV/seleccion_variables_RF_bathandrooms.csv",sep=',')
data_RF = data_RF.drop('Unnamed: 0',axis=1)

columns=['Rooms','Distance','Postcode','Bathroom','Car','Landsize','Propertycount','Distancia_NEW','Longtitude','Lattitude','Location_TRA','Price']
data_RF=data_RF[columns]
#ponemos la columna precio al final

data_RF.head(5)
data_RF.describe()


# No contamos con variables cualitativas. 
# 
# Añadimos una columna adicional categorica para clasificar el precio como hemos visto en el analisis anterior donde dividimos las casa en cara y baratas en funcion de la mediana. 
# 
#     -Casas caras como 1
#     -Casas baratas como 0 

# In[3]:


from sklearn import preprocessing
#Eliminamos columnas que podrían considerarse como categoricas a pesar de ser numericas
#PostCode y PropertyCount

#añadimos una columna adicional categorica para clasificar el precio

intervalos = np.digitize(np.array(data_RF[['Price']]),[875000])
data_RF[['cat_precio']]=intervalos

# convertimos el DataFrame al formato necesario para scikit-learn
data = np.array(data_RF[['Rooms','Distance','Bathroom','Car','Landsize','Distancia_NEW','Longtitude','Lattitude','Price','cat_precio']].values)

y_price=data[:,-1]


# In[4]:



data_RF.dtypes


# El objetivo de este apartado es comprobar como se hace una mejor "Clusterizacion" de los pisos usando KMeans. Vamos a compàrar los resultados de clusterizar usando Kmeans al que se le va a pasar tres variables obtendias mediante TSNE sobre el dataframente original y llevaremos a cabo lo mismo pero empleando las tres variables con mas peso en relacion al precio que se vieron en la practica de de regresion lineal,

# In[5]:


X = data[:,:-2] 
#X = data[:,7:-2]    # nos quedamos con el resto
#X = data[]
feature_names = data_RF.columns[0:-1].to_list()

scaler = preprocessing.StandardScaler().fit(X)
Xs = scaler.transform(X)
data_RF.describe()
y_price

from sklearn.manifold import TSNE

#Take a sample and plot it
N = 5000
random_idx = np.random.choice(Xs.shape[0], N, replace=False)

X_tsne = TSNE(n_components=3, perplexity=20, learning_rate=100,random_state=0).fit_transform(Xs[random_idx,:])
X_new = np.array(data_RF.iloc[:,[7,8,9]].values)
X_new.shape


# Vamos a hacer una vistualización de ambos dataframes, el que tiene tres variables obtenidas mediante TSNE y las tres variables con mas peso del Dataframe original:

# In[6]:



from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

fig = plt.figure(figsize=(6, 6))
ax = Axes3D(fig)
colores=['cyan','orange']
asignar=[]
print(y_price[0])
y_pintar=y_price[random_idx]
print(y_pintar)
for row in y_pintar.astype('int32'):
    asignar.append(colores[row])
ax.set_title('TSNE')
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=asignar,s=40)
ax.legend()

# Pintamos Dataframe original
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

fig = plt.figure(figsize=(6, 6))
ax = Axes3D(fig)
colores=['cyan','orange']
asignar=[]
y_pintar=y_price
for row in y_pintar.astype('int32'):
    asignar.append(colores[row])
ax.set_title('Original Variables')
ax.legend()
ax.scatter(X_new[:, 1], X_new[:, 2], X_new[:, 0], c=asignar,s=40,)


# Como se puede puede ver en las figuras anteriores la relacion con el precio es claramente interpretable en las tres variables directas del dataframe original, no ocurre lo mismo con las variables obtenidas de TSNE. Cabe suponer que si alguno de los dos dataframe nos sirve para hacer clustering que tengan relacion con el precio este será el que llevemos a cabo con el dataframe original. 
# 

# In[7]:


from sklearn.cluster import KMeans


#Inertia para TSNE
K = range(1,20)
inertia = []
for k in K:
    kmeans = KMeans(n_clusters=k).fit(X_tsne)
    inertia.append(kmeans.inertia_)
    
plt.plot(K,inertia,'.-')
plt.title('Cálculo de número de cluster óptimo para dataframe TSNE ')
plt.xlabel('# of clusters')
plt.ylabel('Inertia')
plt.show()



#Inertia para Dataframe original
K = range(1,20)
inertia = []
for k in K:
    kmeans = KMeans(n_clusters=k).fit(X_new)
    inertia.append(kmeans.inertia_)
    
plt.plot(K,inertia,'.-')
plt.title('Cálculo de número de cluster óptimo para dataframe Original ')
plt.xlabel('# of clusters')
plt.ylabel('Inertia')
plt.show()


# Vamos a crear el cluster en primer lugar ambos cluster uno con TSNE y el otro con variables originales y optimizaremos el numero de agrupaciones en cada uno de ellos.

# In[8]:


k_TSNE = 7
kmeans_TSNE = KMeans(n_clusters=k_TSNE, random_state=0)
labels_km_TSNE = kmeans_TSNE.fit_predict(X_tsne)
print("Cluster sizes k-means: {}".format(np.bincount(labels_km_TSNE)))

distances_TSNE = []
for c in kmeans_TSNE.cluster_centers_:
    d = np.sum( np.sum((X_tsne - c) ** 2, axis=1) ) 
    distances_TSNE.append(d.round(2))
    
print("Cluster distances k-means: {}".format(distances_TSNE))

plt.figure(figsize=(12,4))
plt.subplot(121)
plt.bar(range(k_TSNE),np.bincount(labels_km_TSNE))
plt.subplot(122)
plt.bar(range(k_TSNE),distances_TSNE)
plt.show()
plot_silhouette(X_tsne,k_TSNE,kmeans_TSNE.labels_,kmeans_TSNE.cluster_centers_)


# 
# En el caso de TSNE parece que **la mejor agrupacióon nos la hace con K =7 ( 7 clusters)**.
# 
# Pasamos a hacer los mismo con las variables del Dataframe original.

# In[9]:


k = 3
kmeans = KMeans(n_clusters=k, random_state=0)
labels_km = kmeans.fit_predict(X_new)

print("Cluster sizes k-means: {}".format(np.bincount(labels_km)))

distances = []
for c in kmeans.cluster_centers_:
    d = np.sum( np.sum((X_new - c) ** 2, axis=1) ) 
    distances.append(d.round(2))
    
print("Cluster distances k-means: {}".format(distances))

plt.figure(figsize=(12,4))
plt.subplot(121)
plt.bar(range(k),np.bincount(labels_km))

plt.subplot(122)
plt.bar(range(k),distances)
plt.show()

plot_silhouette(X_new,k,kmeans.labels_,kmeans.cluster_centers_)


# 
# En el caso de la agrupación variables originales parece que **la mejor agrupacióon nos la hace con K =3 ( 3 clusters)**.
# 
# Una vez creados los cluster vamos estudiar qué contiene cada uno de ellos

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(6, 6))
ax = Axes3D(fig)

colores=['yellow','red','blue']
asignarx=[]

for row in labels_km.astype('int32'):
    asignarx.append(colores[row])
ax.scatter(X_new[:, 1], X_new[:, 2], X_new[:, 0], c=asignarx,s=20)


# Como se puede puede parece que los tres cluster llevados a cabo han tenido en cuenta perfectamente las variables por lo que cabe espera que la relacion con el precio sea buena.
# 

# In[11]:


data_RF['cluster']=labels_km
sb.pairplot(data_RF.dropna(), hue='cluster',size=4,vars=["Lattitude","Distancia_NEW","Longtitude","cat_precio"],kind='scatter')


# In[12]:



print("elementos mas cercanos y mas lejanos del cluster 0")
close_to_far_from_center(X_new,kmeans.cluster_centers_[0])


# Vamos a ver las variables estadisticas mas importante de cada cluster:

# In[13]:


data_RF.groupby('cluster')["cat_precio","Price"].describe()


# **El cluster 2 los mas baratos** ya que mas del 75 % de los elementos de ese cluster están clasificados como baratos. en el otro extremo está el **cluster 0 donde mas del 50% de los pisos son caros**
# 

#  
#  **pasamos a hacer la misma comprobacion pero para el dataframe extraido de TSNE**

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(6, 6))
ax = Axes3D(fig)

colores=['yellow','red','blue','green','purple','orange','black']
asignarx_tsne=[]

for row in labels_km_TSNE.astype('int32'):
    asignarx_tsne.append(colores[row])
ax.scatter(X_tsne[:, 1], X_tsne[:, 2], X_tsne[:, 0], c=asignarx_tsne,s=20)


# In[15]:


data_RF_xs=data_RF.iloc[random_idx]
data_RF_xs['cluster']=labels_km_TSNE

sb.pairplot(data_RF_xs.dropna(), hue='cluster',size=4,vars=["Lattitude","Distancia_NEW","Longtitude","cat_precio"],kind='scatter')


# In[16]:


print("elementos mas cercanos y mas lejanos del cluster 0")
close_to_far_from_center(X_tsne,kmeans_TSNE.cluster_centers_[0])


# In[17]:


print("elementos mas cercanos y mas lejanos del cluster 1")
close_to_far_from_center(X_tsne,kmeans_TSNE.cluster_centers_[1])


# In[18]:


data_RF_xs.groupby('cluster')["cat_precio","Price"].describe()


# En este caso los cluster 0, 2 y 6 son los que parecen tener los precios mas bajos aunque  el grupo 0 y el 6 ya contienen valores mas altos

# **CONCLUSIÓN: en ambos casos es que en ningun de los dos casos agrupa de manera adecuada al precio (independientemente de si usamos o no TSNE para transformar los datos)**
# 
# Vamos a añadir una variable mas que tenia mucho peso que es el numero de habitaciones y probaremos de nuevo el cluster

# In[19]:


X_2 = np.array(data_RF.iloc[:,[0,7,8,9]].values)
X_2.shape




#Inertia para Dataframe original
K = range(1,20)
inertia = []
for k in K:
    kmeans = KMeans(n_clusters=k).fit(X_2)
    inertia.append(kmeans.inertia_)
    
plt.plot(K,inertia,'.-')
plt.title('Cálculo de número de cluster óptimo para dataframe Original ')
plt.xlabel('# of clusters')
plt.ylabel('Inertia')
plt.show()


# In[20]:


k = 2
kmeans = KMeans(n_clusters=k, random_state=0)
labels_km = kmeans.fit_predict(X_2)

print("Cluster sizes k-means: {}".format(np.bincount(labels_km)))

distances = []
for c in kmeans.cluster_centers_:
    d = np.sum( np.sum((X_2 - c) ** 2, axis=1) ) 
    distances.append(d.round(2))
    
print("Cluster distances k-means: {}".format(distances))

plt.figure(figsize=(12,4))
plt.subplot(121)
plt.bar(range(k),np.bincount(labels_km))

plt.subplot(122)
plt.bar(range(k),distances)
plt.show()

plot_silhouette(X_2,k,kmeans.labels_,kmeans.cluster_centers_)


# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(6, 6))
ax = Axes3D(fig)

colores=['yellow','red','blue']
asignarx=[]

for row in labels_km.astype('int32'):
    asignarx.append(colores[row])
ax.scatter(X_2[:, 1], X_2[:, 2], X_2[:, 0], c=asignarx,s=20)


# In[22]:


data_RF['cluster']=labels_km
sb.pairplot(data_RF.dropna(), hue='cluster',size=4,vars=["Rooms","Lattitude","Distancia_NEW","Longtitude","cat_precio"],kind='scatter')


# In[23]:


print("elementos mas cercanos y mas lejanos del cluster 0")
close_to_far_from_center(X_2,kmeans.cluster_centers_[0])


# In[24]:


data_RF.groupby('cluster')["cat_precio","Price"].describe()

