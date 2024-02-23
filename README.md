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
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: M.CHANDRU
RegisterNumber:  212222230026
*/
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df=pd.read_csv('MLCSV.csv')
df.head()
```
```python
df.tail()
```
```python
X= df.iloc[:,0:-1].values
X
```
```python
Y = df.iloc[:,1].values
Y
```
```python
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
```
```python
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
```
```python
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```
```python
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs scores (test set)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```
```python
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```
## Output:
![Screenshot 2024-02-23 082717](https://github.com/chandrumathiyazhagan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393023/cf358a6d-b8db-4eca-a202-a47ad5cb4c49)
![Screenshot 2024-02-23 082724](https://github.com/chandrumathiyazhagan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393023/860716ce-33ab-4bc8-a96c-6c7e6ac8d8be)
![Screenshot 2024-02-23 082733](https://github.com/chandrumathiyazhagan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393023/80b38ef6-fd25-4a74-91c2-37963bb17faa)
![Screenshot 2024-02-23 082845](https://github.com/chandrumathiyazhagan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393023/c2caf94a-083d-4da4-8e1a-460675df7d7b)
![Screenshot 2024-02-23 082918](https://github.com/chandrumathiyazhagan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393023/9267f523-bdf9-4397-a9cd-c937264d1770)
![Screenshot 2024-02-23 081606](https://github.com/chandrumathiyazhagan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393023/632f01d8-a3f2-493f-ae7a-afc4948f1358)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
