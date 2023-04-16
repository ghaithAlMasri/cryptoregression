import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from handling import handling







if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)
else:
    df = pd.DataFrame()



with st.sidebar:
    st.image('./logo-no-background.png')
    st.title('Easy ML Model Generator')
    choice = st.radio('Please go through the following steps sequentially:', ['Upload Data', 'Profile Data', 'Model Data', 'Download Model!'])
    st.info('The easiest way to getting your pkl for Crpto/Stock regression problems! Download the historical data free at https://www.dukascopy.com/swiss/english/marketwatch/historical/. This was built using Streamlit, Pycaret, Pandas and more!')
    st.write('@2023 GHAITH ALL RIGHTS RESERVED')



            

handling(choice, df)
