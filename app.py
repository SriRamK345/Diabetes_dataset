import numpy as np
import pickle
import streamlit as st


# Load the model
with open("G:/Guvi/VS code/.venv/Diabetic/trained_model.sav", "rb") as file:
    loaded_model = pickle.load(file)


# creatng a function for predection

def diabetes_predection(input_data):

    # changing input to array
    input_to_array = np.asarray(input_data)

    # reshape the array 
    input_reshape = input_to_array.reshape(1 , -1)

    prediction = loaded_model.predict(input_reshape)

    if prediction[0]== 0:
        return "The Person is not diabetic"
    else:
        return "The person is diabetic"

def main():

    # tittle
    st.title("Diabetic Predection")

     # getting input from user 

    Pregnancies = st.text_input("No of Preganacies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("Blood pressure value")
    SkinThickness = st.text_input("Skin Thichness level")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    Age = st.text_input("Age")

    # code for predection
    diagnosis = ""
    # cresting button for predection
    if st.button("Diabetic test result"):
        diagnosis = diabetes_predection([Pregnancies,Glucose,BloodPressure,SkinThickness,
                                            Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    st.success(diagnosis)


if __name__=="__main__":
    main()