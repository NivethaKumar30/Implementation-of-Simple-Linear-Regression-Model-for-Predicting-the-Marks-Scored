### Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored


## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe.
4. Plot the required graph both for test data and training data.
5. Find the values of MSE , MAE and RMSE.

## Program:
```
NAME: NIVETHS .K
REG NO: 212222230102
DEPT:AI&DS
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/studentscores.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
m=lr.coef_
m[0]
b=lr.intercept_
b
```

## Output:

![Screenshot 2024-02-23 104854](https://github.com/NivethaKumar30/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559844/a924f842-8960-4a30-80c3-5d6ade7708c7)

![Screenshot 2024-02-23 104904](https://github.com/NivethaKumar30/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559844/6be722f8-490d-4bb0-9887-f09dc5031f6f)

![Screenshot 2024-02-23 104922](https://github.com/NivethaKumar30/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559844/b7981ed0-6e22-4d00-82d1-f0ecdeda93cf)

![Screenshot 2024-02-23 104934](https://github.com/NivethaKumar30/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559844/2f28de89-d730-4b7a-9770-98da4e0b7c43)

![Screenshot 2024-02-23 104904](https://github.com/NivethaKumar30/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559844/973e6545-ded1-4730-bca4-2d8be8f305bc)

![Screenshot 2024-02-23 104940](https://github.com/NivethaKumar30/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559844/d7fa1ced-6971-47ce-b011-a573ca15d490)

![Screenshot 2024-02-17 030334](https://github.com/NivethaKumar30/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559844/a08496b8-bc54-4336-a409-3c53adae857d)

![Screenshot 2024-02-23 104843](https://github.com/NivethaKumar30/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559844/80b20c2b-4a8e-423e-8a98-5f62f360ffb6)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
