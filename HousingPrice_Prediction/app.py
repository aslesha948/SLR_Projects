#This is an app using Streamlit for the prediction

import streamlit as st
import numpy as np
import pickle

#Loading the Saved Model
model = pickle.load(open(r'/Users/asleshakamera/Projects/SpyderWork/SLR_WORKSHOP/HousingPrice_Prediction/slr_model-housePrices.pkl','rb'))

#Set the title of the Strealit App
st.title("House Prediction App")

#Add a Brief Description
st.write('This app predicts the price of the house depending on the sqare feet of the house using Simple Linear Regression Model')

#Add Input Widget for User to enter the square feet
square_feet = st.number_input("Enter the Square Ft of the house : ",min_value=0.0, max_value=50000.0, value=100.0, step=10.0)

#Button Operation
if st.button("Predict the Price of the House"):
    #Make Prediction using the Trained Model
    size_input = np.array([[square_feet]]) # Convert the input to a 2D array for prediction
    pred = model.predict(size_input)
    
    #Display the Result
    st.success(f"The predicted Price for {square_feet} square feet is :${pred[0]:,.2f}")
    
#Display Information about the model
st.write("The model was trained using a dataset of housing prices with other attributes. It was only trained with one feature.\n Model built by Aslesha")

    
    