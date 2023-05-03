import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Carga los datos
df = pd.read_csv('dataframe/Migration_2010_to_2019.csv')

# Preprocesamiento de los datos
df['population'] = (df['population'] - df['population'].min()) / (df['population'].max() - df['population'].min()) * 100
df['from_different_state_Total'] = (df['from_different_state_Total'] - df['from_different_state_Total'].min()) / (df['from_different_state_Total'].max() - df['from_different_state_Total'].min()) * 100

# Selecciona las variables predictoras
X = df[['population', 'same_house', 'same_state', 'abroad_Total']]

# Selecciona la variable target
Y = df['from_different_state_Total']

# Divide los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=37)

# Entrena el modelo de regresión lineal múltiple
reg = LinearRegression().fit(X_train, y_train)

# Imprime los coeficientes del modelo
print('Coeficientes:', reg.coef_)
print('Intercepto:', reg.intercept_)

# Realiza predicciones sobre los datos de entrenamiento y prueba
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

# Grafica las predicciones sobre los datos de entrenamiento y prueba
plt.scatter(y_train, y_train_pred, color='blue', label='Entrenamiento')
plt.scatter(y_test, y_test_pred, color='red', label='Prueba')
plt.xlabel('Valores reales')
plt.ylabel('Predicciones')
plt.title('Regresión lineal múltiple')
plt.legend(loc='upper left')
plt.show()
