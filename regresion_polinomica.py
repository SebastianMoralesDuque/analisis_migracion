import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

def funcion(X, Y):
    """
    Realiza regresión polinómica de grado 3 y muestra los resultados y gráficos.

    Args:
        X (DataFrame): Matriz de características de tamaño (m, n), donde m es el número de muestras y n es el número de características.
        Y (Series): Vector de respuesta de tamaño (m,).

    Returns:
        None
    """

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.14, random_state=42)

    # Ajustar el modelo de regresión polinómica de grado 3
    poly = PolynomialFeatures(degree=3)
    X_poly_train = poly.fit_transform(X_train)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly_train, Y_train)

    # Hacer predicciones en el conjunto de prueba
    X_poly_test = poly.transform(X_test)
    Y_pred = poly_reg.predict(X_poly_test)

    # Calcular las métricas de evaluación del modelo
    r2 = r2_score(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)

    # Imprimir las métricas de evaluación del modelo
    print('Resultados regresion polinomica\n')
    print("Coeficiente de determinación R^2:", r2)
    print("Error cuadrático medio (MSE):", mse)
    print("Error absoluto medio (MAE):", mae)
    print("Raíz del error cuadrático medio (RMSE):", rmse)

    # Imprimir los coeficientes de la regresión polinómica
    print("Intercept:", poly_reg.intercept_)
    print("Coefficients:", poly_reg.coef_)
    print("\n")

    # Graficar los resultados
    plt.scatter(X_train['population'], Y_train, color='blue', label='Entrenamiento')
    plt.scatter(X_test['population'], Y_test, color='red', label='Prueba')
    plt.legend()

    # Graficar la curva
    X_plot = np.linspace(0, 100, 100).reshape(-1, 4)
    X_plot_poly = poly.transform(X_plot)
    Y_plot = poly_reg.predict(X_plot_poly)
    plt.plot(X_plot[:, 0], Y_plot, color='green')

    plt.title("Migracion dentro de estados unidos (Regresión polinómica de grado 3)")
    plt.xlabel("Población")
    plt.ylabel("Migración desde otros estados")
    plt.show()
