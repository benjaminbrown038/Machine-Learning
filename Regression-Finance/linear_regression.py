
# importing depencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from datetime import datetime

# importing data
df = pd.read_csv('GE_stock_data.csv')

df.reset_index(inplace=True)
x = df.index.values
y = df['Adj Close'].values
a = x[:,None]
b = y[:,None]
z = np.arange(0,250,1)
z=z[:,None]

# fitting model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(a,b)
w = model.predict(z)

df['Predicted'] = w


# plotting financial data vs model
#   1. plotting financial data
plt.plot(df['index'],df['Adj Close'])
plt.plot(df['index'],df['Predicted'])

plt.title('Price vs Time')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
# time.sleep(5)
