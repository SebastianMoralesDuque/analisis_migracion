import minimos_cuadrados
import regresion_simple
import gradiente_descendente
import regresión_multiple
import multiprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


if __name__ == '__main__':

    df = pd.read_csv('dataframe/Migration_2010_to_2019.csv')
    df['population'] = (df['population'] - df['population'].min()) / (df['population'].max() - df['population'].min()) * 100
    df['from_different_state_Total'] = (df['from_different_state_Total'] - df['from_different_state_Total'].min()) / (df['from_different_state_Total'].max() - df['from_different_state_Total'].min()) * 100
    X = df['population'].values.reshape(-1, 1)
    Y = df['from_different_state_Total'].values.reshape(-1, 1)

    p1 = multiprocessing.Process(target=minimos_cuadrados.funcion,args=(X,Y))
    p2 = multiprocessing.Process(target=regresion_simple.funcion,args=(X,Y))
    p3 = multiprocessing.Process(target=gradiente_descendente.funcion,args=(X,Y))


    p1.start()
    p2.start()
    p3.start()


    p1.join()
    p2.join()
    p3.join()


