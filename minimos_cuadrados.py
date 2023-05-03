import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

def funcion(X,Y):

    n = len(X)
    sum_x = np.sum(X)
    sum_y = np.sum(Y)
    sum_xy = np.sum(X * Y)
    sum_x2 = np.sum(X ** 2)

    b1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b0 = np.mean(Y) - b1 * np.mean(X)

    Y_pred = b0 + b1 * X

    # Imprime los resultados
    print(f"MINIMOS CUADRADOS/n")
    print(f"Coeficientes de la regresión lineal: a = {b0:.2f}, b = {b1:.2f}")
    print(f"Error cuadrático medio (MSE): {mean_squared_error(Y, Y_pred):.2f}")
    print(f"Coeficiente de determinación R^2: {r2_score(Y, Y_pred):.2f}")

    plt.scatter(X, Y, color='blue')
    plt.plot(X, Y_pred, color='red')
    plt.title('Migración dentro de Estados Unidos (Minimos Cuadrados)')
    plt.xlabel('Población')
    plt.ylabel('Migrantes desde otros estados')
    plt.show()
