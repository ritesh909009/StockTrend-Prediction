from pyexpat import model
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt 
import pandas_datareader as data
from keras.models import load_model
import streamlit as st


start='2010-01-01'
end = '2022-07-30'

st.title('Closing Stock Trend Prediction')


user_input = st.text_input('Enter The Stock Ticker' , 'AAPL')
df = data.DataReader(user_input , 'yahoo' , start , end)

st.subheader('Data from 2010 - 2022')
st.write(df.describe())


st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100 Days Moving Avg')
MA100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(MA100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100 & 200 Days Moving Avg')
MA100 = df.Close.rolling(100).mean()
MA200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(MA100)
plt.plot(MA200)
plt.plot(df.Close)
st.pyplot(fig)


# Split data
data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.80) : int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_train_array = scaler.fit_transform(data_train)

# Loading model
model = load_model('keras_model.h5')

#Testing part
# past 100 day data

past_100_days = data_train.tail(100)
final_df = past_100_days.append(data_test , ignore_index = True)
input_data = scaler.fit_transform(final_df)

X_test = []
Y_test = []

for i in range(100,input_data.shape[0]):
    X_test.append(input_data[i-100 : i])
    Y_test.append(input_data[i,0])
    

X_test , Y_test = np.array(X_test), np.array(Y_test)
Y_predicted = model.predict(X_test)
scaler=scaler.scale_

scale_factor = 1/scaler[0]
Y_predicted = Y_predicted * scale_factor
Y_test = Y_test * scale_factor



# Final Graph
st.subheader('Prediction VS Original')
fig2=plt.figure(figsize=(12,10))
plt.plot(Y_test , 'b' , label = 'Original Price')
plt.plot(Y_predicted , 'r' , label = 'Predicted Price')
plt.xlabel('Time')
plt.ylebel('Price')
plt.legend()
st.pyplot(fig2)