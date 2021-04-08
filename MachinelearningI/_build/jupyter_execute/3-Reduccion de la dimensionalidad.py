#!/usr/bin/env python
# coding: utf-8

# # 6. Técnicas de reduccion de la dimensionalidad
# 
# Seleccionamos una de las técnias de reduccion de la dimensionalidad para observar como influyen en los modelos. 
#  
# ### PCA 
# 
# En primer lugar,  visuliazamos las correlación entre las difrentes variables par eliminar las que son rependintes 

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm

# Gráficos
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import style

# Preprocesado y modelado

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale


# In[2]:


import warnings
warnings.filterwarnings('ignore')
datos = pd.read_csv('seleccion_variables_RF_bathandrooms.csv')


# In[3]:


datos=datos.drop(columns=['Unnamed: 0'])
datos=datos.drop(columns=['Propertycount'])
datos=datos.drop(columns=['Postcode'])

corr_matrix=datos.corr(method='pearson')         
max_corr=corr_matrix['Price'].sort_values(ascending=False)
datos=datos.drop(columns=['Price'])


# In[4]:


for i in datos.columns:
    if(isinstance(datos[str(i)].iloc[0], ( np.int64))  or isinstance(datos[str(i)].iloc[0],(np.float64))):
        corr_matrix=datos.corr(method='pearson')         
        max_corr=corr_matrix[str(i)].sort_values(ascending=False)
        print('los maximos que correlan con '+str(i)+" son: "+str(max_corr))
    else:
        datos=datos.drop(columns=[str(i)])


# Después de este análisis de correlación vemos que debemos eliminar los siguientes variables: Bathroom' , 'Distance' , 'Car' y 'BathsAndRooms'.

# In[5]:


datos=datos.drop(columns=['Bathroom','Distance','Car','BathsAndRooms'])

datos.mean(axis=0)

import pdb;pdb.set_trace()
datos = datos.apply (pd.to_numeric, errors='coerce')

datos = datos.dropna()
datos.reset_index(drop=True)

print('-------------------------')
print('Varianza de cada variable')
print('-------------------------')
datos.var(axis=0)


# Entrenamiento modelo PCA con escalado de los datos julia  datos=datos.drop(columns=['Longtitude','Lattitude'])

# In[ ]:


pca_pipe = make_pipeline(StandardScaler(), PCA())
pca_pipe.fit(datos)
import pdb;pdb.set_trace()

# Se extrae el modelo entrenado del pipeline
modelo_pca = pca_pipe.named_steps['pca']

import pdb;pdb.set_trace()

# Se combierte el array a dataframe para añadir nombres a los ejes.
componentes_df=pd.DataFrame(data    = modelo_pca.components_,columns = datos.columns,index   = ['PC1', 'PC2', 'PC3', 'PC4','PC5'])

import pdb;pdb.set_trace()


# Realizamos una visualizacion de las componentes y decidimos cual borrar 

# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
componentes = modelo_pca.components_
plt.imshow(componentes.T, cmap='viridis', aspect='auto')
plt.yticks(range(len(datos.columns)), datos.columns)
plt.xticks(range(len(datos.columns)), np.arange(modelo_pca.n_components_) + 1)
plt.grid(False)
plt.colorbar();


# Explicamos el porcentaje de varianza explicada por cada componente.

# In[ ]:


print('----------------------------------------------------')
print('Porcentaje de varianza explicada por cada componente')
print('----------------------------------------------------')
print(modelo_pca.explained_variance_ratio_)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.bar(
    x      = np.arange(modelo_pca.n_components_) + 1,
    height = modelo_pca.explained_variance_ratio_
)

for x, y in zip(np.arange(len(datos.columns)) + 1, modelo_pca.explained_variance_ratio_):
    label = round(y, 2)
    ax.annotate(
        label,
        (x,y),
        textcoords="offset points",
        xytext=(0,10),
        ha='center'
    )

ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
ax.set_ylim(0, 1.1)
ax.set_title('Porcentaje de varianza explicada por cada componente')
ax.set_xlabel('Componente principal')
ax.set_ylabel('Por. varianza explicada');
import pdb;pdb.set_trace()


# Utilizaremos estas PCA en los modelos a continuación. 
