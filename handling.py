import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pycaret.regression import setup, compare_models, pull, save_model, load_model


class handling:
    def __init__(self, choice, df):
        self.choice = choice
        self.df = df
        self.handle_choice()


    def handle_choice(self):
        if self.choice == 'Upload Data':
            st.title('Welcome to Crypto Predictor!')
            st.subheader('Upload your Crypto Data 📈')
            data = st.file_uploader('test', label_visibility="hidden")
            if data:
                self.df = pd.read_csv(data)
                self.df = self.df[self.df.Volume>0]
                self.handle_date()
                self.df.to_csv('dataset.csv', index=None)
                st.dataframe(self.df)

        if self.choice == 'Profile Data':
            st.subheader('Lets Make Some Analysis 🔬')
            self.profile = ProfileReport(self.df, title="Pandas Profiling Report")
            st_profile_report(self.profile)

        if self.choice == 'Model Data':
            st.subheader('Lets Create The Predictor 🤖')
            self.chosen_target = st.selectbox('Select Your Target Column: ', self.df.columns)
            if st.button('Start Training'):
                st.write('Beep Boop... ⚙️')
                setup(self.df, target=self.chosen_target)
                self.setup_df = pull()
                st.dataframe(self.setup_df)
                self.best_model = compare_models()
                self.compare_df = pull()
                st.dataframe(self.compare_df)
                save_model(self.best_model, 'best_model')



        if self.choice == 'Download Model!':
            st.subheader('Get The Best Performing Model! 🔥')
            with open('best_model.pkl', 'rb') as f: 
                st.download_button('Download Model 🚀', f, file_name="best_model.pkl")



    def handle_date(self):
        for index,dates in enumerate(self.df['Local time']):
            self.df['Local time'][index] = dates.split()[0]