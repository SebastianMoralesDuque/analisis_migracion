import minimos_cuadrados
import regresion_simple
import gradiente_descendente
import regresion_multiple
import regresion_polinomica
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    # Lectura del archivo CSV con los datos de migración
    df = pd.read_csv('dataframe/Migration_2010_to_2019.csv')

    # Exploración inicial de los datos
    print("Exploración inicial de los datos:\n")
    print(df.head())   # Muestra los primeros 5 datos del DataFrame
    print("\n")
    print(df.columns)  # Muestra los nombres de las columnas del DataFrame
    print("\n")

    # Normalización de las variables de población y migración
    df['population'] = (df['population'] - df['population'].min()) / (df['population'].max() - df['population'].min()) * 100
    df['from_different_state_Total'] = (df['from_different_state_Total'] - df['from_different_state_Total'].min()) / (df['from_different_state_Total'].max() - df['from_different_state_Total'].min()) * 100
    
    # Separación de los datos de entrada (X) y los datos de salida (Y)
    X = df['population'].values.reshape(-1, 1)
    Y = df['from_different_state_Total'].values.reshape(-1, 1)

    # Creación de los procesos para cada tipo de regresión
    p1 = multiprocessing.Process(target=minimos_cuadrados.funcion,args=(X,Y))
    p2 = multiprocessing.Process(target=regresion_simple.funcion,args=(X,Y))
    p3 = multiprocessing.Process(target=gradiente_descendente.funcion,args=(X,Y))

    # Inicio del primer proceso
    p1.start()

    # Espera a que termine el primer proceso antes de iniciar el segundo
    p1.join()

    # Inicio del segundo proceso
    p2.start()

    # Espera a que termine el segundo proceso antes de iniciar el tercero
    p2.join()

    # Inicio del tercer proceso
    p3.start()

    # Espera a que termine el tercer proceso antes de iniciar el cuarto
    p3.join()

    # Selección de variables adicionales para regresión polinómica y múltiple
    X = df[['population', 'same_house', 'same_state', 'abroad_Total']]
    p4 = multiprocessing.Process(target=regresion_multiple.funcion,args=(X,Y))
    p5 = multiprocessing.Process(target=regresion_polinomica.funcion,args=(X,Y))

    # Inicio del cuarto proceso
    p4.start()

    # Espera a que termine el cuarto proceso antes de iniciar el quinto
    p4.join()

    # Inicio del quinto proceso
    p5.start()

    # Espera a que termine el quinto proceso antes de finalizar el programa
    p5.join()

 

print("\nCONCLUSIONES\n")

print("\nEl primer método utilizado fue el de mínimos cuadrados, el cual muestra un coeficiente de determinación R^2 de 0.80 y un error cuadrático medio (MSE) de 87.16. Esto indica que la relación entre la población y el número de personas que migran de diferentes estados tiene cierta correlación positiva, pero la relación no es muy fuerte.\n")

print("El segundo método utilizado fue la regresión simple, que muestra resultados muy similares a los de mínimos cuadrados con un coeficiente de determinación R^2 de 0.79 y un MSE de 97.75.\n")

print("El tercer método utilizado fue el de gradiente descendente, que muestra resultados similares a los de mínimos cuadrados con un coeficiente de determinación R^2 de 0.80 y un MSE de 85.54.\n")

print("El cuarto método aplicado fue el de regresión múltiple, que muestra un coeficiente de determinación R^2 de 0.8329 y un MSE de 21.6. Estos resultados indican que la relación entre la población y el número de personas que migran de diferentes estados tiene una correlación más fuerte cuando se consideran otras variables además de la población.\n")

print("El quinto método utilizado fue el de regresión polinómica, que muestra el coeficiente de determinación R^2 más alto de todos los métodos con 0.929 y un MSE de 21.6. Esto indica que la relación entre la población y el número de personas que migran de diferentes estados es más fuerte cuando se ajusta una curva polinómica a los datos.\n")

print("En conclusión, los resultados sugieren que la relación entre la población y el número de personas que migran de diferentes estados tiene una correlación positiva, pero no muy fuerte. Además, los resultados indican que la relación se fortalece cuando se consideran otras variables y cuando se ajusta una curva polinómica a los datos.\n")


print("Gráficos de dispersión:")
pd.plotting.scatter_matrix(df, figsize=(10, 10))
plt.show()
print("\n")

