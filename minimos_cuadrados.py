import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

def funcion(X, Y):
    """
    Esta función toma dos arreglos de datos X e Y, realiza una regresión lineal
    por mínimos cuadrados entre ellos, y grafica los resultados.

    Parameters:
    X (numpy.ndarray): arreglo unidimensional con los datos del eje X
    Y (numpy.ndarray): arreglo unidimensional con los datos del eje Y

    Returns:
    None
    """
    # Calculamos los valores necesarios para la regresión
    n = len(X) # número de datos
    sum_x = np.sum(X) # suma de los valores en X
    sum_y = np.sum(Y) # suma de los valores en Y
    sum_xy = np.sum(X * Y) # suma de los productos X*Y
    sum_x2 = np.sum(X ** 2) # suma de los cuadrados de X

    # Calculamos los coeficientes de la regresión lineal por mínimos cuadrados
    b1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b0 = np.mean(Y) - b1 * np.mean(X)

    # Calculamos los valores de Y predichos por la regresión
    Y_pred = b0 + b1 * X

    # Imprimimos los resultados de la regresión
    print('Resultados Minimos cuadrados\n')
    print(f"Coeficientes de la regresión lineal: a = {b0:.2f}, b = {b1:.2f}")
    print(f"Error cuadrático medio (MSE): {mean_squared_error(Y, Y_pred):.2f}")
    print(f"Coeficiente de determinación R^2: {r2_score(Y, Y_pred):.2f}")
    print("\n")

    # Graficamos los datos y la línea de regresión
    plt.scatter(X, Y, color='blue')
    plt.plot(X, Y_pred, color='red')
    plt.title('Migración dentro de Estados Unidos (Minimos Cuadrados)')
    plt.xlabel('Población')
    plt.ylabel('Migrantes desde otros estados')
    plt.show()
