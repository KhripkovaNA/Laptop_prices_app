import streamlit as st
import os
import matplotlib.pyplot as plt

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(FILE_DIR, "image", "musk.png")

st.title('Flipkart Laptop Data - Business Insights on Product Pricing')

st.write(''.join(['If you want to start your own laptop company in India, ',
                  'you should know how to estimate the price of laptops your company is going to create.']))
st.write('This app allows you to do that!')
st.markdown('Ilon Musk is welcome to check it out \N{winking face}')
img = plt.imread(IMAGE_PATH)
st.image(img, width=400)
st.caption('Innomatics Research Labs Internship - February 2023')
