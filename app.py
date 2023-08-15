import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from joblib import load
import math

data = pd.read_pickle('data.pkl')

st.title("Laptop Price Predictor")

company = st.selectbox('Brand', data['Company'].unique())

typename = st.selectbox('TypeName', data['TypeName'].unique())

ram = int(st.selectbox('Ram(in GB)', [2, 4, 6, 8, 12, 16, 32, 64]))

os = st.selectbox('Operating System', data['OpSys'].unique())

weight = float(st.number_input('Weight(in Kg)'))

touchscreen = st.selectbox('Touch Screen', ['No', 'Yes'])

ips = st.selectbox('IPS', ['No', 'Yes'])

screen_size = st.number_input('Screen Size')

resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
                                                '2800x1800', '2560x1600','2560x1440'])

cpu = st.selectbox('Processor', data['Processor'].unique())

hdd = int(st.selectbox('HDD(in GB)', [0, 128, 254, 512, 1024, 2056]))

sdd = int(st.selectbox('SDD(in GB)', [0, 8, 16, 64, 128, 254, 512, 1024, 2056]))

gpu = st.selectbox('GPU', data['Gpu Brand'].unique())

predictor = load('predictor.joblib')
if st.button('Predict'):
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = float(((X_res**2) + (Y_res**2))**0.5/screen_size)

    label_encoders = load('label_encoders.joblib')
    scaler = load('scaler.joblib')

    user_inputs = {
        'Company': company,
        'TypeName': typename,
        'OpSys': os,
        'Processor': cpu,
        'Gpu Brand': gpu
    }

    if all(user_inputs.values()):
        # Apply Label Encoding to user inputs
        encoded_inputs = {}

        for col, user_input in user_inputs.items():
            le = label_encoders[col]
            encoded_inputs[col] = (le.transform([user_input])[0])

        inp = [encoded_inputs['Company'], encoded_inputs['TypeName'], ram, encoded_inputs['OpSys'], weight,
               touchscreen, ips, ppi,encoded_inputs['Processor'], sdd, hdd, encoded_inputs['Gpu Brand']]
        inp = np.array([inp])

    y_test = scaler.transform(inp)

    result = math.floor(np.exp(predictor.predict(y_test))[0])
    result_string = f"The predicted value is Rs {result}"
    st.header(result_string)


