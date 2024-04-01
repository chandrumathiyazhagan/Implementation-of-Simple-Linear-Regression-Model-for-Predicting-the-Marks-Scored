# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:

Program to implement the simple linear regression model for predicting the marks scored.

Developed by: M.CHANDRU

RegisterNumber:  212222230026


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/ml.csv')
df.head(10)
```
```python
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
```
```python
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train
y_train
```
```python
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
x_train
y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_train,lr.predict(x_train),color='orange')
```
```python
lr.coef_
```
```python
lr.intercept_
```

## Output:

![Screenshot 2024-04-01 202900](https://github.com/chandrumathiyazhagan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393023/b30b5975-f091-4bff-9eb7-40ad304e5840)

![Screenshot 2024-04-01 202911](https://github.com/chandrumathiyazhagan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393023/b1e19387-365b-4090-ad33-b1c7ed7b75c8)

![Screenshot 2024-04-01 202919](https://github.com/chandrumathiyazhagan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393023/b4942083-e431-4e11-a9da-f978821a8a4c)

![Screenshot 2024-04-01 203006](https://github.com/chandrumathiyazhagan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393023/01b0584a-4c56-4349-b914-0d88173d079c)

![Screenshot 2024-04-01 203029](https://github.com/chandrumathiyazhagan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393023/61b1489e-523c-4b78-b7b3-136cb172762b)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
