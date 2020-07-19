# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 21:30:12 2020

@author: mehaf
"""
#importing the librarires
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the dataset
dataset=pd.read_csv('Salary_data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
#splitting the data set into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
#fitting
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
#predicting
y_pred=regressor.predict(x_test)
#visualizing the training set
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience Training Set')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
#visualizing the test set
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience Test Set')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()