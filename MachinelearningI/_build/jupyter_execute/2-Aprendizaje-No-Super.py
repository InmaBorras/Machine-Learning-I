#!/usr/bin/env python
# coding: utf-8

# # 2.Aprendizaje no supervisado 
# 
# 
# 
# 
# Cargamos las librerías que vamos a necesitar  y el archivo donde se encuetran las funiones que usaremos posteriormente. 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.getcwd()

from src.utils import pd, np, plt

# Useful functions
from src.utils import load_examples, plot_scatter, plot_silhouette


# Cargamos los datos con las variables ya que hemos previamente selecionado. 

# In[2]:


# cargamos los datos
data_RF = pd.read_csv("seleccion_variables_RF_bathandrooms.csv",sep=',')

data_RF = data_RF.drop('Unnamed: 0',axis=1)#eliminamos la primera columna. 

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

# Categorizamos el precio
mediana= 870000.0
data_RF["precio_div"]= np.where(data_RF["Price"]>=mediana, "1", "0" )#1 es caro Y 0 es barato 
print(data_RF.head(20))

#Eliminamos del dataframe las varibales categoricas  ya que no son  relevantes
data_RF=data_RF[['Rooms','Distance','Bathroom','Car','Landsize','Distancia_NEW','Longtitude','Lattitude','Price','precio_div']]

# convertimos el DataFrame al formato necesario para scikit-learn
data = np.array(data_RF[['Rooms','Distance','Bathroom','Car','Landsize','Distancia_NEW','Longtitude','Lattitude','Price','precio_div']].values)
data[:,-1]# aqui no se que hace quitamos la cateogiria creo



y_price_cat=data[:,-1]#seleccionamos solo la variable precio_div
y_price=data[:,-2]
X = data[:,:-2] 
X_selec = data[:,5:-2] # seleccionamos las variables 'Distancia_NEW','Longtitude','Lattitude'
print(y_price_cat)


# Estandarizamos lo datos para poder agruparlos y representarlos mejor. 

# In[4]:


# mas reduccion de la dimensionalidad 
feature_names = data_RF.columns[0:-1].to_list()# quitamos las columnas de precio categorizado 
scaler = preprocessing.StandardScaler().fit(X)
Xs = scaler.transform(X)


# ##  Visualización de los datos 
# 
# Primero reducimos la dimensionalidad  con el TSNE para poder representar todas la variables 
# Tambien probamos selecionado las variables  fueron selecionadas en los  modelos ateriores  "Distania_NEW", " Latitude" y "longitude". 
# 
# 
# "Distania_NEW", "Room" " Latitude", "Landsize" y " Bathrooms". 

# In[5]:


from sklearn.manifold import TSNE

#Cogemos una muestra del dataframe y realizamos un TSNE 
N = 5000
random_idx = np.random.choice(Xs.shape[0], N, replace=False)
X_tsne = TSNE(n_components=2, perplexity=10, learning_rate=100,random_state=0).fit_transform(Xs[random_idx,:])# tecnica de reduccin del todo el data frame se queda con 3 columans 


#Seleccionamos las columnas que mas nos interesan que son las anteriormente indicadas. 
X_new = np.array(data_RF.iloc[:,[5,6,7]].values)
X_new.shape


# In[6]:


data_RF.iloc[1,[5,6,7]].values# esto no se por que lo ha hecho. 


# Visualizacion de los datos con TSNE. Habria que probar diferentes preplexies para ver cual es la que se adapta mejor. 

# In[7]:


print("Visualizacion de los datos por el TSNE")
plt.figure(figsize=(6, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c = 'b', marker='o', alpha=0.2)
plt.xticks([])
plt.yticks([])
plt.show();


#----------------------------
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
fig = plt.figure(figsize=(6, 6))
ax = Axes3D(fig)
colores=['cyan','orange','blue','yellow']
asignar=[]
print(y_price_cat[0])
y_pintar=y_price_cat[random_idx]
print(y_pintar)
for row in y_pintar.astype('int32'):
    asignar.append(colores[row])

ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 1], c=asignar,s=40)
#ax.scatter(X_new[:, 1], X_new[:, 2], X_new[:, 0], c=asignar,s=20)


# Visualizamos con las categorias seleccionadas para ver si tiene sentido o es mas claro. 

# In[8]:


print("Visualizacion de los datos con las categorias seleccionadas")
plt.figure(figsize=(6, 6))
plt.scatter(X_new[:, 0], X_new[:, 1], c = 'b', marker='o', alpha=0.2)
plt.xticks([])
plt.yticks([])
plt.show();


#----------------------------
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
fig = plt.figure(figsize=(6, 6))
ax = Axes3D(fig)
colores=['cyan','orange','blue','yellow']
asignar=[]
print(y_price_cat[0])
y_pintar=y_price_cat
print(y_pintar)
for row in y_pintar.astype('int32'):
    asignar.append(colores[row])

#ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=asignar,s=40)
ax.scatter(X_new[:, 1], X_new[:, 1], X_new[:, 0], c=asignar,s=20)


# Parece que se ven mejor seleccionando las categorias que hemos seleccionado.En el cual, los datos mas caros estan abajo  naranjas y se difunden con los azules amedida que aumentamos al distancia al centro. 
# 
# Duda: las variables estan correlacionadas?? Poner ejemplos de otras varaibles??

# ## K-Means 
# 
# Usaremos este modelo por que es simple y util para dataset grandes además es una técnica escalable. 
# 
# 
# La estrategia que usaremos para realizar K-Means es :
# 
# - Representar `inertia`  para determinar el mejor número de clusters.  
# - Analizar el numero de muestras en cada uno de los clusters y la sumas de las distancias al centroide. 
# - Para cada cluster, `display`  el ejemplo de $n$  mas cercano y el mas lejano  de cada centroide.
# - Analyze the features distribution for each cluster.
# 
# La idea es pasarle el que procede TNSE y las 3 mejores columnas. Comenzamos con  las 3 mejores variables según nuesta seleccion anterior ya que los resultados del TNSE no eran muy especificos. 

# In[9]:


from sklearn.cluster import KMeans

K = range(1,20)

inertia = []
for k in K:
    kmeans = KMeans(n_clusters=k).fit(X_new)
    inertia.append(kmeans.inertia_)
       
print("Inertia acorde a las categorias selecionadas")  

plt.plot(K,inertia,'.-')
plt.xlabel('# of clusters')
plt.ylabel('Inertia')
plt.show()


# La "inertia" corresponde a mas o menos 3 o 4 clusters. Comprobaremos que es mejor posteriormente. 
# Realizamos el algoritmo de K Means, primero probamos con 4 cluster segun hemos visualizado. 

# In[10]:


k = 4 
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
# Parece que los clusters estan bastante descompensados, se puede observar en los diferentes tamaños y distancias, a pesar de que el coeficiende de silueta esta cerca de 0,6 por lo que los puntos de diferentes centroides estan bastante separados. 
# 
# Relizamos la misma prueba pero con 3 clusters para ver como se distribuyen. 

# In[11]:


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


# In[19]:


clusters= pd.DataFrame(labels_km)


#  
# Parece que los clusters estan bastante descompensados, se puede observar en los diferentes tamaños y distancias. Sin embargo, en el coeficiente de silueta vemos que todos soperan el 0.6 por lo que estan bastante separados unos cluster de otros aunque los tamaños no lo sean.
# 
# Podemos conlcuir que la separacion en 3 cluster es mas precisa ya que estan mejor separados entre si los datos de los diferentes clusters.

# In[25]:


##Yo casi que quitaria esto 

#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sb

#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure(figsize=(6, 6))
#ax = Axes3D(fig)

#colores=['cyan','orange','blue','yellow']
#asignarx=[]

#for row in labels_km.astype('int32'):# asiga a cada row rn labels un color diferente  para cada uno de los clusters
    #asignarx.append(colores[row])

#ax.scatter(X_new[:, 1], X_new[:, 2], X_new[:, 0], c=asignarx,s=20)#AQUI parece que esta usadno el resutlado de Kmeans
#data_RF['cluster']=labels_km



#sb.pairplot(data_RF.dropna(), hue='cluster',size=4,vars=["Lattitude","Distancia_NEW","Longtitude","precio_div"],kind='scatter')


# In[20]:


def close_to_far_from_center(X,centroid, n=10):
    
    distance = np.sum((X - centroid) ** 2, axis=1)
    
    print('Close to center')
    display(data_RF.iloc[np.argsort(distance)[:n]])
    
    print('Far from center')
    display(data_RF.iloc[np.argsort(distance)[-n:]])
    


# In[30]:



from scipy import stats
print("Cluster 0")
close_to_far_from_center(X_new,kmeans.cluster_centers_[0])
#stats.describe(kmeans.cluster_centers_[1].sort())


# In[31]:



print("Cluster 1")
close_to_far_from_center(X_new,kmeans.cluster_centers_[1])


# In[32]:



print("Cluster 2")
close_to_far_from_center(X_new,kmeans.cluster_centers_[2])


# A priori, no se puede observar una gran relación entre la categorización que hemos realizado anteriormente con el precio y los cluster seleccionados habría que estudiar mas en profundidad estas relaciones. 

# In[27]:


#feature = 'Price'
#col_number = feature_names.index(feature)

#plt.figure(figsize=(15,10))
#for l in np.unique(labels_km):
    
    #plt.subplot(2,5,l+1)
    #plt.hist(X[labels_km == l,col_number],bins = 50, density=True)
    #plt.xlabel(feature)
    #plt.title('Cluster #' + str(l))

#plt.show()


# ## DBSCAN
# 
# Usamos otra tecnica de clustering mas compleja  para corroborar la infrormacion obtenida anteriormente. 
# 
# eps muy grande muy pocos clusters...
# 

# In[ ]:


#from sklearn.cluster import DBSCAN

#for eps in [1, 3]: # 5, 7]:# a quui a dejado solo el 1 y 3 
    #print("\neps={}".format(eps))
    #dbscan = DBSCAN(eps=eps, min_samples=20)
    #labels = dbscan.fit_predict(Xs)
    #print("Number of clusters: {}".format(len(np.unique(labels))))
    #print("Cluster sizes: {}".format(np.bincount(labels + 1)))


# In[ ]:


#from sklearn.cluster import DBSCAN

#for eps in [1, 3]: 
    #print("\neps={}".format(eps))
    #dbscan = DBSCAN(eps=eps, min_samples=25)
   # labels = dbscan.fit_predict(Xs)
    #print("Number of clusters: {}".format(len(np.unique(labels))))
    #print("Cluster sizes: {}".format(np.bincount(labels + 1)))


# ## Hierarchical clustering

# In[ ]:


#from scipy.cluster.hierarchy import dendrogram, linkage

#Z = linkage(X, 'average')
#dendrogram(Z)
#plt.show()


# In[ ]:




