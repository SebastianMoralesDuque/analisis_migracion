import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Definir la función de costo (MSE)
def cost_function(X, y, coefficients):
    """
    Calcula la función de costo para una regresión lineal.

    Args:
        X (ndarray): Matriz de características de tamaño (m, n+1), donde m es el número de muestras y n es el número de características.
        y (ndarray): Vector de respuesta de tamaño (m,).
        coefficients (ndarray): Vector de coeficientes de tamaño (n+1,).

    Returns:
        float: El valor de la función de costo para los coeficientes dados.
    """
    m = len(y)
    J = np.sum((X.dot(coefficients) - y)**2) / (2*m)
    return J

# Definir la función de actualización de los coeficientes de regresión lineal
def gradient_descent(X, y, coefficients, alpha, num_iterations):
    """
    Realiza el descenso del gradiente para actualizar los coeficientes de una regresión lineal.

    Args:
        X (ndarray): Matriz de características de tamaño (m, n+1), donde m es el número de muestras y n es el número de características.
        y (ndarray): Vector de respuesta de tamaño (m,).
        coefficients (ndarray): Vector de coeficientes de tamaño (n+1,).
        alpha (float): Tasa de aprendizaje.
        num_iterations (int): Número de iteraciones.

    Returns:
        ndarray: El vector de coeficientes optimizados.
        ndarray: Un vector de tamaño (num_iterations,) que contiene los valores de la función de costo en cada iteración.
    """
    m = len(y)
    J_history = np.zeros(num_iterations)
    for i in range(num_iterations):
        h = X.dot(coefficients)
        for j in range(len(coefficients)):
            coefficients[j] = coefficients[j] - alpha*(1/m)*(np.sum((h - y)*X[:,j]))
        J_history[i] = cost_function(X, y, coefficients)
    return coefficients, J_history

def funcion(X, y):
    """
    Realiza una regresión lineal usando el método de gradiente descendente y muestra los resultados.

    Args:
        X (ndarray): Matriz de características de tamaño (m, n), donde m es el número de muestras y n es el número de características.
        y (ndarray): Vector de respuesta de tamaño (m,).

    Returns:
        None
    """
    m = len(y)

    # Añadir una columna de unos a X para el término de intercepción
    X = np.column_stack((np.ones(m), X))

    # Dividir los datos en conjuntos de entrenamiento y prueba (50% para entrenamiento, 50% para prueba)
    train_size = int(0.5 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Entrenar el modelo de regresión lineal con los datos de entrenamiento
    reg = LinearRegression().fit(X_train[:, 1:], y_train)

    # Obtener los coeficientes de regresión lineal
    coefficients = np.append(reg.intercept_, reg.coef_)

    # Calcular el MSE y el R^2 con los datos de prueba
    mse = np.mean((reg.predict(X_test[:, 1:]) - y_test)**2)
    r2 = reg.score(X_test[:, 1:], y_test)

    # Definir los hiperparámetros
    alpha = 0.000000004
    num_iterations = 400

    # Realizar el descenso del gradiente para obtener los coeficientes optimizados
    new_coefficients, J_history = gradient_descent(X, y, coefficients, alpha, num_iterations)

    # Imprimir los resultados
    print('Resultados gradiente descendente\n')
    print('Coeficientes de la regresión lineal:', coefficients)
    print('Error cuadrático medio (MSE):', mse)
    print('Coeficiente de determinación R^2:', r2)
    print("\n")

    # Graficar la función de costo en función del número de iteraciones
    plt.plot(range(num_iterations), J_history)
    plt.xlabel('Número de Iteraciones')
    plt.ylabel('Función de Costo')
    plt.title('Funcion de costo')
    plt.show()

    # Generar la gráfica
    plt.scatter(X[:, 1], y, color='blue')
    plt.plot(X[:, 1], new_coefficients[0] + new_coefficients[1]*X[:, 1], color='red')
    plt.xlabel('Veces Jugado')
    plt.ylabel('Número de Reseñas')
    plt.title('Migración dentro de Estados Unidos (Gradiente descendente)')
    plt.show()