# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.import the needed packages.
2.Assigning hours to x and scores to y.
3.Plot the scatter plot.
4.Use mse,rmse,mae formula to find the values.
```

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SASHMITHA SREE K V
RegisterNumber:  212224230255/24900551
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv("/content/student_scores (2).csv")  # Load the dataset
print(df)                               # Print the full DataFrame

df.head(0)                              # Show header only
df.tail(0)                              # Show header only

print(df.head())                        # Display first 5 rows
print(df.tail())

x = df.iloc[:, :-1].values              # Extracting Hours (features)
print(x)

y = df.iloc[:, 1].values                # Extracting Scores (target)
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=1/3, random_state=0
)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print("Predicted values:", y_pred)
print("Actual values:", y_test)

plt.scatter(x_train, y_train, color='black')
plt.plot(x_train, regressor.predict(x_train), color='yellow')
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test, y_test, color='black')
plt.plot(x_train, regressor.predict(x_train), color='pink')  # line stays the same
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse = mean_absolute_error(y_test, y_pred)
print('MSE =', mse)

mae = mean_absolute_error(y_test, y_pred)
print('MAE =', mae)

rmse = np.sqrt(mse)
print("RMSE =", rmse)

```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)

![image](https://github.com/user-attachments/assets/d1ef831d-0d37-4999-b11b-503449f4a704)

![image](https://github.com/user-attachments/assets/33eae06b-3bcd-41a1-81e1-99d077283ab9)

![image](https://github.com/user-attachments/assets/1d2cb9f3-c3a5-4043-b72f-3324ff582033)

![image](https://github.com/user-attachments/assets/f4768c0f-428b-438f-a78b-c3ad16967de7)

![image](https://github.com/user-attachments/assets/49691117-d841-4d32-a6cd-d3414023ecb9)

![image](https://github.com/user-attachments/assets/b0ab9b5f-b741-48f8-873d-11f855141175)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
