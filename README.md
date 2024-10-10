# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1 : Start

STEP 2 : Load the dataset, convert the feature columns and target column to float, and apply standard scaling to normalize the data.

STEP 3 : Add a bias term (column of ones) to the feature matrix and initialize the parameter vector theta to zeros.

STEP 4 : For a fixed number of iterations, compute predictions, calculate the errors, and update the parameter vector theta using the gradient descent formula.

STEP 5 : Pass the scaled features and target values to the linear_regression function to learn the optimal theta values.

STEP 6 : Scale new data using the same scaling technique, then predict the target value by performing matrix multiplication with the learned parameters theta.

STEP 7 : Inverse transform the predicted scaled value to get the original-scale prediction.

STEP 8 : Display the predicted target value for the new data point.
STEP 9 : End

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Manoj MV
RegisterNumber:  212222220046
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        #calculate predictions
        predictions=(X).dot(theta).reshape(-1,1)
        
        #calculate erros
        errors=(predictions-y).reshape(-1,1)
        
        #update theta using gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data = pd.read_csv("C:/Users/SEC/Downloads/50_Startups.csv",header=None)
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#Learn model Parameters
theta=linear_regression(X1_Scaled,Y1_Scaled)
#Predict target value for a new data point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```

## Output:
Head Values

![image](https://github.com/user-attachments/assets/16e6232f-54ef-4059-9525-89faa427688c)

Predicted Value

![image](https://github.com/user-attachments/assets/5342906b-25da-437e-91de-4764a9b3a56c)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
