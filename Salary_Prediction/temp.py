#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 09:31:48 2025

@author: asleshakamera
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'/Users/asleshakamera/Desktop/DSANDML/Salary_Data.csv')

#Split the data into two different variables

x= dataset.iloc[:,:-1]
y= dataset.iloc[:,-1]

#Assigning test size as 20%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
#Assigning test size as 30%
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()#regressor is the model.
regressor.fit(x_train, y_train)

#Predicting the result

y_pred = regressor.predict(x_test)

#Plot the Best fit liine
plt.scatter(x_test, y_test, color ='red')#real Salary
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Test SET)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# # Slope of the model
# m_slope = regressor.coef_
# print(m_slope)
# # Intercept of the model
# c_intercept = regressor.intercept_
# print(c_intercept)

# pred_12yr_emp_exp = m_slope * 12 + c_intercept
# print(pred_12yr_emp_exp)

# pred_20yr_emp_exp = m_slope * 20 + c_intercept
# print(pred_20yr_emp_exp) 7


bias = regressor.score(x_train,y_train)
print(bias) #Training part #high bias, low variance :Underfitting. 

variance =regressor.score(x_test, y_test)
print(variance)

#Stats for ML
dataset.mean()

dataset['Salary'].mean()

#Median
dataset['Salary'].median()

#Mode
dataset['Salary'].mode()

#variance
dataset.var()

dataset['Salary'].var()

#Standard Deviation
dataset.std()
##Coef of variation
from scipy.stats import variation

variation(dataset.values)


#correlation
dataset.corr()

#Sum of Squared Error,SSR
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE
y=y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

#SST
mean_total = np.mean(dataset.values)
SST  = np.sum((dataset.values-mean_total)**2)
print(SST)

#r2
r_square = 1-SSR/SST
print(r_square)

#Deployment in flask and Html

import pickle

#Save the trained model to disk
filename = 'linear_regression_model.pk1'

#Open a file in write-binaru model
with open(filename,'wb') as file:
    pickle.dump((regressor), file)
    
print('Model has been pickled and saved as linear_regression_model.pk1 ')

import os
os.getcwd()







