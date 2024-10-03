import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_excel('BI_Clientes09-1.xlsx')

# Preprocess the data
features = ['HouseOwnerFlag', 'NumberCarsOwned', 'CommuteDistance', 'Region', 'Age']
X = pd.get_dummies(data[features])  # Convert categorical variables to dummy variables
y = data['BikeBuyer']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Visualize the decision tree
# plt.figure(figsize=(12, 8))
# plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
# plt.title('Decision Tree for Bike Buyer Prediction')
# plt.show()

y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report

# Calcular precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión: {accuracy}')

# Generar un reporte de clasificación
print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

# Crear un nuevo DataFrame con los datos del nuevo cliente
nuevo_cliente = pd.DataFrame({
    'HouseOwnerFlag': [1],
    'NumberCarsOwned': [2],
    'CommuteDistance': ['0-1 Miles'],
    'Region': ['North'],
    'Age': [35]
})

# Convertir las variables categóricas en variables dummy
nuevo_cliente_dummies = pd.get_dummies(nuevo_cliente)

# Alinear las columnas con el conjunto de entrenamiento
nuevo_cliente_dummies = nuevo_cliente_dummies.reindex(columns=X.columns, fill_value=0)

# Hacer la predicción
prediccion = clf.predict(nuevo_cliente_dummies)
print(f'El nuevo cliente {"comprará" if prediccion[0] else "no comprará"} una bicicleta.')
