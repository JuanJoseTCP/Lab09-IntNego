import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

# Cargar los datos desde el archivo Excel
data = pd.read_excel('BI_Postulantes09-1.xlsx')

# Preprocesar los datos: seleccionar las características relevantes
features = [
    'Apertura Nuevos Conoc.',
    'Nivel Organización',
    'Participación Grupo Social',
    'Grado Empatía',
    'Grado Nerviosismo',
    'Dependencia Internet'
]
X = data[features]

# Aplicar K-Means clustering
k = 3  # Elegir el número de clusters
kmeans = KMeans(n_clusters=k, random_state=42)  # Inicializar el modelo de K-Means
data['Cluster'] = kmeans.fit_predict(X)  # Ajustar el modelo y predecir los clusters

# Visualizar los clusters generados
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data, x='Apertura Nuevos Conoc.', y='Nivel Organización', hue='Cluster', palette='viridis', s=100)
plt.title('K-Means Clustering de Postulantes')
plt.xlabel('Apertura Nuevos Conoc.')
plt.ylabel('Nivel Organización')
plt.legend(title='Cluster')
plt.show()

# Generar histogramas para visualizar relaciones entre clusters y especialidades
plt.figure(figsize=(12, 8))
for i, feature in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sns.histplot(data=data, x=feature, hue='Cod_Especialidad', multiple='stack', bins=10)
    plt.title(f'Histograma de {feature} por Especialidad')
plt.tight_layout()
plt.show()
