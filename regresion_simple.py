import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def funcion(x,y):
    #Dividir los datos en conjunto de entrenamiento y de prueba
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=37)
    """
    test_size: la proporción del conjunto de datos que se utilizará para la prueba. Si se establece en 0.2, el 20% 
    del conjunto de datos se utilizará para la prueba y el 80% se utilizará para el entrenamiento.
    random_state: se utiliza para inicializar el generador de números aleatorios que realiza la división aleatoria.
    Si se establece en un número entero, el resultado de la división aleatoria será el mismo cada vez que se ejecute el código.
    """

    # Crear el modelo de regresión lineal
    reg_model = LinearRegression()

    # Ajustar el modelo a los datos de entrenamiento
    reg_model.fit(x_train, y_train)

    # Coeficientes de la regresión lineal
    a = reg_model.intercept_[0]
    b = reg_model.coef_[0][0]

    # Predecir los valores de y utilizando los datos de prueba
    y_pred = reg_model.predict(x_test)

    # Error cuadrático medio (MSE) utilizando los datos de prueba
    mse = mean_squared_error(y_test, y_pred)

    # Coeficiente de determinación R^2 utilizando los datos de prueba
    r2 = r2_score(y_test, y_pred)

    print('Resultados regresion simple\n')
    print("Coeficientes de la regresión lineal: a = {:.2f}, b = {:.2f}".format(a, b))
    print("Error cuadrático medio (MSE): {:.2f}".format(mse))
    print("Coeficiente de determinación R^2: {:.2f}".format(r2))
    print("\n")



    # Crear la figura y los ejes
    fig, ax = plt.subplots()

    # Graficar los puntos de datos de prueba
    ax.scatter(x_test, y_test, color='blue')

    # Graficar la regresión lineal utilizando los datos de prueba
    ax.plot(x_test, y_pred, color='red')

    # Configurar los ejes y las etiquetas
    plt.title('Migración dentro de Estados Unidos (Regresion simple)')
    plt.xlabel('Población')
    plt.ylabel('Migrantes desde otros estados')

    # Mostrar la gráfica
    plt.show()
