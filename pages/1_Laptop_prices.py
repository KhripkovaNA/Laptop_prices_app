import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import re

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
DATA_PATH = os.path.join(PARENT_DIR, "data", "laptop.csv")


@st.cache_data
def load_dataset(data_link):
    dataset = pd.read_csv(data_link)
    return dataset


df = load_dataset(DATA_PATH)
X = df.drop('MRP', axis=1)
y = df['MRP']
X[['RAM_size', 'Drive_size', 'SSD_size']] = np.log2(X[['RAM_size', 'Drive_size', 'SSD_size']]+1)

preprocessor = ColumnTransformer(
    [
        ('onehot', OneHotEncoder(handle_unknown='ignore'), ['CPU_type', 'OS', 'RAM_type']),
        ('scaler', StandardScaler(), ['CPU_cores', 'RAM_size', 'SSD_size', 'Drive_size'])
    ],
    remainder='passthrough'
)
X_pr = preprocessor.fit_transform(X)
rfr = RandomForestRegressor(random_state=87)
rfr.fit(X_pr, np.log(y))

st.header("Laptop Price Estimation")
cpu_type = st.selectbox('Select Processor Brand', ['Intel', 'AMD', 'Apple', 'Other'])
cpu_cores = st.slider('Select Number of Processor Cores', 2, 128, 4)
check = st.checkbox('I don`t know number of processor cores')
if check:
    cpu = st.text_input('Input Processor Name', 'Core i5')
    conditions = [bool(re.search('Dual|i3', cpu)),
                  bool(re.search('Quad|i5', cpu)),
                  bool(re.search('Hexa|i7|i9', cpu)),
                  bool(re.search('Octa|M1$|M1 Pro', cpu)),
                  bool(re.search('M2|M1 Max', cpu))]
    cores = [2, 4, 6, 8, 10]
    cpu_cores = int(np.select(conditions, cores, default=4))
    st.write('Estimated Number of Cores is ' + str(cpu_cores))
ram_type = st.selectbox('Select RAM Type',
                        ['DDR4', 'DDR5', 'LPDDR4X', 'Unified Memory',
                         'LPDDR5', 'LPDDR4', 'LPDDR3', 'Other'])
if re.search('DDR[3|4]', ram_type):
    ram_type = 'DDR4'
elif re.search('DDR5', ram_type):
    ram_type = 'DDR5'
ram_size = st.slider('Select RAM Capacity in GB', 2, 128, 2, step=2)
drive_type = st.multiselect('Select Storage Type', ['HDD', 'SSD', 'eMMC'])

if 'HDD' in drive_type:
    hdd = 1
    hdd_size = st.number_input(f'Input HDD Capacity in GB', 0, value=512)
else:
    hdd = 0
    hdd_size = 0
if 'SSD' in drive_type:
    ssd = 1
    ssd_size = st.number_input(f'Input SSD Capacity in GB', 0, value=512)
else:
    ssd = 0
    ssd_size = 0
if 'eMMC' in drive_type:
    emmc = 1
    emmc_size = st.number_input(f'Input eMMC Capacity in GB', 0, value=64)
else:
    emmc = 0
    emmc_size = 0
drive_size = hdd_size + ssd_size + emmc_size
os = st.selectbox('Select Operating System', ['Windows', 'Mac OS', 'DOS', 'Chrome', 'Other'])
X_test = pd.DataFrame({'CPU_type': [cpu_type], 'CPU_cores': [cpu_cores],
                       'OS': [os], 'RAM_size': [ram_size],
                       'RAM_type': [ram_type], 'SSD_size': [ssd_size],
                       'SSD': [ssd], 'Drive_size': [drive_size]})

X_test[['RAM_size', 'Drive_size', 'SSD_size']] = np.log2(X_test[['RAM_size', 'Drive_size', 'SSD_size']]+1)
X_test_pr = preprocessor.transform(X_test)
st.divider()
if st.button('Predict laptop price'):
    y_pr = round(np.exp(rfr.predict(X_test_pr))[0])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption('Processor Brand')
        st.write('**' + cpu_type + '**')
    with col2:
        st.caption('Number of Cores')
        st.write('**' + str(cpu_cores) + '**')
    with col3:
        st.caption('Operating System')
        st.write('**' + os + '**')
    col11, col12, col13 = st.columns(3)
    with col11:
        st.caption('RAM Type')
        st.write('**' + ram_type + '**')
    with col12:
        st.caption('RAM Capacity')
        st.write('**' + str(ram_size) + 'GB' + '**')
    if ssd == 1:
        st.caption('SSD Capacity')
        st.write('**' + str(ssd_size) + 'GB' + '**')
    if hdd == 1:
        st.caption('HDD Capacity')
        st.write('**' + str(hdd_size) + 'GB' + '**')
    if emmc == 1:
        st.caption('eMMC Capacity')
        st.write('**' + str(emmc_size) + 'GB' + '**')
    st.subheader('Predicted price is ' + str(y_pr) + '₹')
