# ==============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import style

# Preprocesado y modelado
# ==============================================================================
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

# Configuración warnings
# ==============================================================================  Unnamed: 0
import warnings
warnings.filterwarnings('ignore')
datos = pd.read_csv('./seleccion_variables_RF_bathandrooms.csv')


#datos["BathsAndRooms"]=(datos["Rooms"]+datos["Bathroom"])/datos["Distancia_NEW"].apply(np.log10)
#datos.to_csv('seleccion_variables_RF_bathandrooms.csv', index=False)


datos=datos.drop(columns=['Unnamed: 0'])
datos=datos.drop(columns=['Propertycount'])
datos=datos.drop(columns=['Postcode'])

corr_matrix=datos.corr(method='pearson')         
max_corr=corr_matrix['Price'].sort_values(ascending=False)
datos=datos.drop(columns=['Price'])

for i in datos.columns:
    if(isinstance(datos[str(i)].iloc[0], ( np.int64))  or isinstance(datos[str(i)].iloc[0],(np.float64))):
        corr_matrix=datos.corr(method='pearson')         
        max_corr=corr_matrix[str(i)].sort_values(ascending=False)
        print('los maximos que correlan con '+str(i)+" son: "+str(max_corr))
    else:
        datos=datos.drop(columns=[str(i)])

#despues de este analisis de correlacion vemos que debemos eliminar los siguientes   datos=datos.drop(columns=['Longtitude'])

#MIRAR TAMBIEN CON BathsAndRooms
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

# ==============================================================================
pca_pipe = make_pipeline(StandardScaler(), PCA())
pca_pipe.fit(datos)
import pdb;pdb.set_trace()

# Se extrae el modelo entrenado del pipeline
modelo_pca = pca_pipe.named_steps['pca']

import pdb;pdb.set_trace()

# Se combierte el array a dataframe para añadir nombres a los ejes.
componentes_df=pd.DataFrame(data    = modelo_pca.components_,columns = datos.columns,index   = ['PC1', 'PC2', 'PC3', 'PC4','PC5'])

import pdb;pdb.set_trace()




# ==============================================================================
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
componentes = modelo_pca.components_
plt.imshow(componentes.T, cmap='viridis', aspect='auto')
plt.yticks(range(len(datos.columns)), datos.columns)
plt.xticks(range(len(datos.columns)), np.arange(modelo_pca.n_components_) + 1)
plt.grid(False)
plt.colorbar();

#componentes_df.var(axis=0)




# Porcentaje de varianza explicada por cada componente
# ==============================================================================
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
