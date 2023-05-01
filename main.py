import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Migration_2010_to_2019.csv')
X = df['population'].values.reshape(-1, 1)
Y = df['from_different_state_Total'].values.reshape(-1, 1)

reg = LinearRegression()
reg.fit(X, Y)

b0 = reg.intercept_[0]
b1 = reg.coef_[0][0]

Y_pred = reg.predict(X)

plt.scatter(X, Y, color='blue')
plt.plot(X, Y_pred, color='red')
plt.title('Migración dentro de Estados Unidos')
plt.xlabel('Población')
plt.ylabel('Migrantes desde otros estados')
plt.show()
