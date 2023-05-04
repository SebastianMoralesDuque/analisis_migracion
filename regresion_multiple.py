import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from mpl_toolkits.mplot3d import Axes3D

def funcion(X, Y):
    """
    Realiza una regresión lineal múltiple con regularización Lasso sobre los datos de migración dentro de Estados Unidos.

    Args:
        X (DataFrame): Matriz de características de tamaño (m, n), donde m es el número de muestras y n es el número de características.
        Y (Series): Vector de respuesta de tamaño (m,).

    Returns:
        None
    """

    # Divide los datos en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)

    # Entrenar el modelo de regresión lineal con regularización Lasso
    reg = Lasso(alpha=0.1).fit(X_train, y_train)

    # Realizar predicciones y calcular el R2
    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)
    print('Resultados regresion multiple\n')
    print('R2 (entrenamiento):', r2_score(y_train, y_train_pred))
    print('R2 (prueba):', r2_score(y_test, y_test_pred))

    # Imprime los coeficientes del modelo
    print('Coeficientes:', reg.coef_)
    print('Intercepto:', reg.intercept_)
    print("\n")

    # Graficar la relación entre las características y la respuesta
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train['population'], X_train['same_house'], y_train, c='blue', label='Entrenamiento')
    ax.scatter(X_test['population'], X_test['same_house'], y_test, c='red', label='Prueba')
    ax.set_xlabel('Población')
    ax.set_ylabel('Misma casa')
    ax.set_zlabel('Desde otro estado')
    ax.set_title('Migración dentro de Estados Unidos (Regresion Multiple)')
    ax.legend(loc='upper left')
    plt.show()
