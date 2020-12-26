#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

st.header("Предсказание цены на дом")
st.write("""
Введите данные для предсказания релевантной цены вашего дома
""")


# In[14]:


def user_input_features():
    bedrooms = st.number_input("Общее количество комнат", value=1, min_value=0, step=1)
    bathrooms = st.number_input("Количество ванных комнат", value=1, min_value=0, step=1)
    sqft_living = st.number_input("Общая ЖИЛАЯ площадь дома, футы в квадрате", value=100, min_value=0, step=1)
    sqft_lot = st.number_input("Общая площадь дома, футы в квадрате", value=100, min_value=0, step=1)
    floors = st.number_input("Количество этажей в доме", value=1, min_value=1, step=1)
    waterfront = st.selectbox('Вид на водопад? 1 - да, 0 - нет', (1, 0))
    grade = st.slider('Оценка дома', 1, 11, 0)

    data = {'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft_living': sqft_living,
            'sqft_lot': sqft_lot,
            'floors': floors,
            'waterfront': waterfront,
            'grade': grade,
            }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()


# In[16]:


#Write out input selection
st.subheader('Входные данные (Pandas DataFrame)')
st.write(input_df)

#Load in model
load_clf = pickle.load(open('house_model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(input_df)

st.subheader('Релевантная цена за дом, $')
st.write(int(prediction))


# In[ ]:




