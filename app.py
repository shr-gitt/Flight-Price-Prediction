import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#load the saved model
model_path='/Users/mandipshrestha/Desktop/Flight-Price-Prediction/ML_Model/random_forest_regression.pkl'
model=joblib.load(model_path)
price_scaler = joblib.load('ML_MODEL/price_scaler.pkl')

expected_columns=['Airline_Air Asia', 'Airline_Air India', 'Airline_GoAir','Airline_IndiGo', 'Airline_Jet Airways', 'Airline_Jet Airways Business','Airline_Multiple carriers','Airline_Multiple carriers Premium economy', 'Airline_SpiceJet','Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy','Source_Banglore', 'Source_Chennai', 'Source_Delhi', 'Source_Kolkata','Source_Mumbai', 'Destination_Banglore', 'Destination_Cochin','Destination_Delhi', 'Destination_Hyderabad', 'Destination_Kolkata','Destination_New Delhi', 'Total_Stops', 'Date', 'Month','Dep_hours', 'Dep_min', 'Arrival_hours', 'Arrival_min', 'Duration']
scaler = StandardScaler()
def main():
    #Set the title of the web app
    st.title('Price Prediction')
    
    #Add a description
    st.write('Enter flight information to predict price.')
    
    #Create columns for layout
    col1,col2=st.columns([2,1])
    
    with col1:
        st.subheader('Flight Information')
        
        #Add input fields for features
        Airline=st.selectbox('Airline Name',['IndiGo','Air India','Jet Airways','SpiceJet','Multiple carriers','GoAir''Vistara','Air Asia','Vistara Premium economy','Jet Airways Business','Multiple carriers Premium economy','Trujet'])
        Source=st.selectbox("Source",['Banglore','Kolkata','Delhi','Chennai','Mumbai'])
        Destination=st.selectbox("Destination",['New Delhi','Banglore','Cochin','Kolkata','Delhi','Hyderabad'])
        Total_Stops=st.slider("Total Stops",0,4)
        Date=st.slider('Date',1,30)
        Month=st.slider('Month',1,12)
        Dep_hours=st.slider('Departure hour',0,23,5)
        Dep_min=st.slider('Departure minute',0,59,15)
        Duration_hours=st.slider('Duration hour',0,23,2)
        Duration_min=st.slider('Duration minute',0,59,15)
        
        Arrival_min=Dep_min+Duration_min
        if(Arrival_min>59):
            Arrival_min=Arrival_min-59
            z=1
        else:
            z=0
        Arrival_hours=Dep_hours+Duration_hours+z
        if(Arrival_hours>23):
            Arrival_hours=Arrival_hours-23
            z=1
        st.write(f'Arrival hour:{Arrival_hours}')
        st.write(f'Arrival minute:{Arrival_min}')
        
    
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'Airline_Air Asia':[1 if Airline=='Air Asia' else 0],
        'Airline_Air India':[1 if Airline=='Air India' else 0],
        'Airline_GoAir':[1 if Airline=='GoAir' else 0],
        'Airline_IndiGo':[1 if Airline=='IndiGo' else 0],
        'Airline_Jet Airways':[1 if Airline=='Jet Airways' else 0],
        'Airline_Jet Airways Business':[1 if Airline=='Jet Airways Business' else 0],
        'Airline_Multiple carriers':[1 if Airline=='Multiple carriers' else 0],
        'Airline_Multiple carriers Premium economy':[1 if Airline=='Multiple carriers Premium economy' else 0],
        'Airline_SpiceJet':[1 if Airline=='SpiceJet' else 0],
        'Airline_Trujet':[1 if Airline=='Trujet' else 0],
        'Airline_Vistara':[1 if Airline=='Vistara' else 0],
        'Airline_Vistara Premium economy':[1 if Airline=='Vistara Premium economy' else 0],
        'Source_Banglore':[1 if Source=='Banglore' else 0],
        'Source_Chennai':[1 if Source=='Chennai' else 0],
        'Source_Delhi':[1 if Source=='Delhi' else 0],
        'Source_Kolkata':[1 if Source=='Kolkata' else 0],
        'Source_Mumbai':[1 if Source=='Mumbai' else 0],
        'Destination_Banglore':[1 if Destination=='Banglore' else 0],
        'Destination_Cochin':[1 if Destination=='Cochin' else 0],
        'Destination_Delhi':[1 if Destination=='Delhi' else 0],
        'Destination_Hyderabad':[1 if Destination=='Hyderabad' else 0],
        'Destination_Kolkata':[1 if Destination=='Kolkata' else 0],
        'Destination_New Delhi':[1 if Destination=='New Delhi' else 0],
        'Total_Stops':[Total_Stops],
        'Date':[Date],
        'Month':[Month],
        'Dep_hours':[Dep_hours],
        'Dep_min':[Dep_min],
        'Arrival_hours':[Arrival_hours],
        'Arrival_min':[Arrival_min],
        'Duration':[Duration_hours*60+Duration_min],
    })

    # Ensure columns are in the same order as during model training
    input_data = input_data[expected_columns]

    # Prediction and results section
    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            prediction = model.predict(input_data)
            #probability = model.predict_proba(input_data)[0][1]
            rfm_predictions = price_scaler.inverse_transform(prediction.reshape(-1, 1))
            st.write(f'Prediction for Price: {rfm_predictions}')
            #st.write(f'Probability of Passing: {probability:.2f}')

            # Plotting
            fig, axes = plt.subplots(3, 1, figsize=(8, 16))
if __name__ == '__main__':
    main()
