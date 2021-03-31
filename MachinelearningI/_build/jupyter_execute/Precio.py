#!/usr/bin/env python
# coding: utf-8

# # Categorización por Precio
# 
# En primer luegar analizamos la distribución del precio.  

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import statistics as stat
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns

data= pd.read_csv("/home/inma/Master_Data_Science _Contenido/Machine_learningI/Practica/Machine-Learning-I/Seleccion_variables.csv")


# In[2]:


data.head()


# In[3]:


precio=data[["Price"]]
print(precio.describe())
mediana= precio.median()
print("la mediana de" , mediana)
#moda=(stat.mode(precio)
moda=st.mode(precio)    
print("La moda de  Price" , moda)


# In[4]:


sns.histplot(data=precio, kde=True, stat='density')# poner la raya de la mediana 


# In[5]:


data.sort_values("Price")
mediana= 870000.0
precio_bajo= data[data["Price"]< mediana]
precio_alto= data[data["Price"]>= mediana]


# In[6]:


print(len(precio_bajo))
print(len(precio_alto))


# In[ ]:





# 
# ```{toctree}
# :hidden:
# :titlesonly:
# 
# 
# markdown
# notebooks
# ```
# 
