#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pickle
model = pickle.load(open('./heart_disease_ML.pkl', 'rb'))



# In[5]:


def predict(age, sex, cp, trestbps, chol, thalach,exang, oldpeak, slope, ca, thal):
    prediction=model.predict([[age, sex, cp, trestbps, chol, thalach, exang,oldpeak, slope,ca, thal]])
    return prediction

def main():
    st.title("HEART DISEASE PREDICTION")
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit HEART DISEASE PREDICTION</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    age = st.text_input(label="age",placeholder="Enter age in number")
    sex = st.text_input(label="Sex",placeholder="Enter 1 for Male and 0 for Female")
    cp = st.text_input(label="Chest pain type",placeholder="Enter 0 for typical angina, 1 for atypical angina, 2 for non-anginal pain, 3 for asymptomatic")
    trestbps = st.text_input(label="Resting Blood Pressure",placeholder="Enter in numbers unit is mm Hg")
    chol=st.text_input(label="Serum Cholestrol",placeholder="Enter in number mg/dl")
    thalach=st.text_input(label="maximum heart rate",placeholder="Enter in number")
    exang=st.text_input(label="Exercise induced angina",placeholder="Enter 1 for yes enter 0 for no")
    oldpeak=st.text_input(label="ST depression induced by exercise",placeholder="Enter number")
    slope=st.text_input(label='the slope of the peak exercise ST segment',placeholder="0 for upsloping, 1 for flat, 2 for downsloping")
    ca=st.text_input(label="number of vessels", placeholder="Enter value between 0 and 3")
    thal=st.text_input(label="thal",placeholder="0 for normal, 1 for fixed defect, 2 for reversable defect")
    result=""
    if st.button("Predict"):
        result=predict(age, sex, cp, trestbps, chol, thalach,exang,oldpeak, slope, ca, thal)
    if result==0:
        st.success('You don\'t have a heart condition')
    elif result==1:
        st.success('You have a heart condition')
    else:
        st.success("waiting for input")
    
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
main()


