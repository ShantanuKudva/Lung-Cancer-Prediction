import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open("lung_model.sav", 'rb'))

#creating the function for prediction
def lung_prediction(input_data):
    # change the data to a numpy array
    input_data_as_numpy_array = np.array(input_data, dtype=float)

    # reshape the numpy array as we are prediction for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)


    if (prediction[0] == 0):
        return("The person could have low levels of Lung Cancer")
    elif (prediction[0] == 1):
        return("The person could have medium levels of Lung Cancer")
    else:
        return("The person could have higher levels of Lung cancer")


def main():
    #Giving the title for the app
    st.title('Lung Cancer Prediction Web App')

    #getting the input data from the user
    age=st.text_input("Age of the patient")
    sex = st.text_input("sex (1 = male; 2 = female)")
    pollution=st.text_input("The level of air pollution exposure of the patient. (1-8)")
    alcohol=st.text_input("The level of alcohol use of the patient. (1-8)")
    dust=st.text_input("The level of dust allergy of the patient. (1-8)")
    occupation=st.text_input("The level of occupational hazard of the patient. (1-8)")
    genes=st.text_input("The level of genetic risk of the patient. (1-7)")
    cld=st.text_input("The level of chronic lung disease of the patient. (1-7)")
    diet = st.text_input("The level of balanced diet of the patient. (1-7)")
    obesity=st.text_input("The level of obesity of the patient. (1-7)")
    smoke=st.text_input("The level of smoking of the patient. (1-8)")
    passive =st.text_input("The level of passive smoker of the patient. (1-8)")
    cp=st.text_input("The level of chest pain of the patient. (1-9)")
    blood=st.text_input("The level of coughing of blood of the patient. (1-9)")
    fatigue = st.text_input("The level of fatigue of the patient. (1-9)")
    wl=st.text_input("The level of weight loss of the patient. (1-8)")
    sob=st.text_input("The level of shortness of breath of the patient. (1-9)")
    wheeze=st.text_input("The level of wheezing of the patient. (1-8)")
    swallow=st.text_input("The level of swallowing difficulty of the patient. (1-8)")
    club=st.text_input("The level of clubbing of finger nails of the patient. (1-9)")
    cold=st.text_input("The frequency of cold of the patient. (1-7)")
    dryCough=st.text_input("The level of clubbing of finger nails of the patient. (1-7)")
    snoring=st.text_input("The level of snoring of the patient. (1-7)")
    #code for prediction
    diagnosis=''

    #creating a button for prediction
    if st.button("Lung Cancer Test Result"):
        diagnosis=lung_prediction([age,sex,pollution,alcohol,dust,occupation,genes,cld,diet,obesity,smoke,passive,cp,blood,fatigue,wl,sob,wheeze,swallow,club,cold,dryCough,snoring])

    st.success(diagnosis)

if __name__ == "__main__":
    main()
