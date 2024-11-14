# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the data, plot a scatter plot of population (in tens of thousands) vs. profit, and set labels and title.
2. Define the cost function computeCost to calculate the squared error for predictions.
3. Prepare X with a column of ones and the population data, y as the profit data, and initialize theta.
4. Define gradientDescent to update theta iteratively, minimizing the cost function, and store cost history (J_history).
5. Run gradientDescent with chosen parameters, print the hypothesis function, and plot the cost function over iterations.
6. Plot the regression line with predicted profits, and use the predict function to estimate profits for populations of 35,000 and 70,000.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: KALAISELVAN J
RegisterNumber: 212223080022
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01, num_iters=1000):
     X=np.c_[np.ones(len(X1)), X1]
     theta=np.zeros (X.shape[1]).reshape(-1,1)
     for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
     return theta

data=pd.read_csv("50_Startups.csv" ,header = None) 
X=(data.iloc[1:, :-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot (np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print (data.head)
print(f"Predicted value: {pre}") 
*/
```

## Output:
<img width="1050" alt="Screenshot 2024-10-25 at 9 30 51 AM" src="https://github.com/user-attachments/assets/2c2c5775-3140-43d7-b40a-5e466406d3e4">
<img width="1011" alt="Screenshot 2024-10-25 at 9 31 38 AM" src="https://github.com/user-attachments/assets/a8b088bc-9c59-498c-be1b-cfe48acc0fb4">

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
