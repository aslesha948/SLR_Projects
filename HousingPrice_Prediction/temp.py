import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#np.set_printoptions(threshold=np.nan)

dataset = pd.read_csv(r'/Users/asleshakamera/Projects/SpyderWork/SLR_WORKSHOP/HousingPrice_Prediction/House_data.csv')
#Since price is a dependent variable. Let's first check what are the features of the dataset
dataset.columns
#reshape(-1, 1) is a convenient way to convert a one-dimensional array into a two-dimensional array with a single column, allowing for easier manipulation and compatibility with various algorithms and functions.
x = np.array(dataset['sqft_living']).reshape(-1,1)
y = np.array(dataset['price'])

#Assigning the test_size as 30% and spliting the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Predicting the Prices
y_pred = regressor.predict(x_test)

#Visualising the test set  Results or Plotting the Best Fit line
plt.scatter(x_test,y_test,color = 'red') #Real Prices
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Sqft-living against the Price(Testing Set)')
plt.xlabel('Sqft-Living')
plt.ylabel('Price of the House')
plt.show()

#Visualising the Training set  Results or Plotting the Best Fit line
plt.scatter(x_train,y_train,color = 'green') #Real Prices
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title('Sqft-living against the Price(Training Set)')
plt.xlabel('Sqft-Living')
plt.ylabel('Price of the House')
plt.show()

import os 
os.getcwd()
target_directory = '/Users/asleshakamera/Projects/SpyderWork/SLR_WORKSHOP/HousingPrice_Prediction'
#Deployment in Flask and HTML
import pickle

#Saving the trained model to Disk
filename = os.path.join(target_directory,'slr_model-housePrices.pkl')

#Open a File in write-binary model
with open(filename,'wb') as file:
    pickle.dump((regressor),file)

print('Model has been pickled and saved as slr_model-housePrices.pkl')



