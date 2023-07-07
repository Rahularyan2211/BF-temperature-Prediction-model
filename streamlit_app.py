import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('BF_data.csv', index_col=[0])
x_vars = df.drop(['SAT_1', 'SAT_2', 'SAT_3', 'SAT_4'], axis=1)
x_vars.drop('DATE_TIME', axis=1, inplace=True)
y_vars = df[['SAT_1', 'SAT_2', 'SAT_3', 'SAT_4']]
x_train, x_test, y_train, y_test = train_test_split(
    x_vars, y_vars, test_size=0.3,  random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

dnn_model = joblib.load('dnn_model.pkl')
rnd_forest_model = joblib.load('random_forest.pkl')
multiple_linear_reg_model = joblib.load('multiple_linear_regression.pkl')


st.title('Prediction of Temperature Parameters of Blast Furnace')
st.write("""
         In this app, Fill up the necessary input parameters to obtain the Average Skin Temperature over the course of next 4 hours """)

cold_blast_flow = st.number_input("Cold Blast Flow", value=311727)
cold_blast_pressure = st.number_input("Cold Blast Pressure", value=3.15)
cold_blast_temperature = st.number_input("Cold Blast Temperature", value=129)
steam_flow = st.number_input("Steam Flow", value=4)
steam_temperature = st.number_input("Steam Temperature", value=213)
steam_pressure = st.number_input("Steam Pressure", value=3.34)
o2_pressure = st.number_input("O2 Pressure", value=3.2)
o2_flow = st.number_input("O2 Flow", value=7296)
o2_percentage = st.number_input("O2 Percentage", value=23.08)
pci = st.number_input("Pulverized Coal Injection", value=32)
atm_humdity = st.number_input("Atmospheric Humdity", value=24.56)
hot_blast_temperature = st.number_input("Hot Blast Temperature", value=1060)
hot_blast_pressure = st.number_input("Hot Blast Pressure", value=2.99)
top_gas_pressure = st.number_input("Top Gas Pressure", value=1.5)
top_gas_temp_after_10 = st.number_input(
    "Top Gas Temperature after 10 minutes", value=112)
top_gas_temp_after_20 = st.number_input(
    "Top Gas Temperature after 20 minutes", value=135)
top_gas_temp_after_30 = st.number_input(
    "Top Gas Temperature after 30 minutes", value=107)
top_gas_temp_after_40 = st.number_input(
    "Top Gas Temperature after 40 minutes", value=130)
top_gas_spray = st.number_input("Top Gas Spray", value=0)
top_gas_temp = st.number_input("Top Gas Temperature", value=121)
top_gas_pressure_1 = st.number_input("Top Gas Pressure 1", value=2)
co = st.number_input("CO (carbon monoxide)", value=22.22)
co2 = st.number_input("CO2 (carbon dioxide)", value=21)
h2 = st.number_input("H2 (Hydrogen)", value=3.88)
avg_skin_temp = st.number_input("Average Skin Temperature", value=69.940478)
option = st.selectbox(
    'Select the learning algorithm:',
    ('Multiple Linear Regression', 'Random Forest Regression', 'Deep Neural Network Regression'))


def predict():
    row = np.array([cold_blast_flow, cold_blast_pressure, cold_blast_temperature, steam_flow, steam_temperature, steam_pressure, o2_pressure, o2_flow, o2_percentage, pci, atm_humdity, hot_blast_temperature,
                   hot_blast_pressure, top_gas_pressure, top_gas_temp_after_10, top_gas_temp_after_20, top_gas_temp_after_30, top_gas_temp_after_40, top_gas_spray, top_gas_temp, top_gas_pressure_1, co, co2, h2, avg_skin_temp])
    row = scaler.transform([row])
    X = pd.DataFrame(row, columns=x_vars.columns.to_list())
    if option == 'Multiple Linear Regression':
        prediction = multiple_linear_reg_model.predict(X)
    elif option == 'Random Forest Regression':
        prediction = rnd_forest_model.predict(X)
    else:
        prediction = dnn_model.predict(X)
    st.write("""
    # Predicted temperatures for next 4 hours:""")
    st.write('Average Skin Temperature after 1 Hour: ',
             round(prediction[0][0], 4), "°C")
    st.write('Average Skin Temperature after 2 Hours: ',
             round(prediction[0][1], 4), "°C")
    st.write('Average Skin Temperature after 3 Hours: ',
             round(prediction[0][2], 4), "°C")
    st.write('Average Skin Temperature after 4 Hours: ',
             round(prediction[0][3], 4), "°C")


trigger = st.button('Predict', on_click=predict)
cols = x_vars.columns.to_list()
