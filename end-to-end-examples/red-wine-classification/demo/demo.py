# %%
from operator import concat
import streamlit as st
import requests
import os


# %%
def fetch(session, url):
    try:
        result = session.get(url)
        return result.json()
    except Exception:
        return {}

# %%

mini = [4.600000,0.120000,0.000000,0.900000,0.012000,1.000000,6.000000,0.990070,2.740000,0.330000,8.400000]
maxi = [15.900000,1.580000,1.000000,15.500000,0.611000,72.000000,289.000000,1.003690,4.010000,2.000000,14.900000]
# %%
from pyexpat import features
from wsgiref.util import request_uri


def main():
    request_url = os.getenv("INFERENCE_SERVER_URL")
    st.set_page_config(page_title="Trial", page_icon="🤖")
    st.title("Wine Quality Prediction")
    session = requests.Session()
    with st.form("my_form"):
        
        fixed_acidity = st.slider("Fixed Acidity",min_value=0.0,max_value=16.0,step=0.1,key="fixed_acidity")
        volatile_acidity = st.slider("Volatile Acidity",min_value=0.0,max_value=1.6,step=0.01,key="volatile_acidity")
        citric_acid = st.slider("Citric Acid",min_value=mini[2],max_value=maxi[2],step=0.01)
        residual_sugar = st.slider("Residual Sugar",min_value=mini[3],max_value=maxi[3],step=0.1)
        chlorides = st.slider("Chlorides",min_value=mini[4],max_value=maxi[4],step=0.001)
        free_sulfur_dioxide = st.slider("Free Sulfur Dioxide",min_value=mini[5],max_value=maxi[5],step=0.1)
        total_sulfur_dioxide = st.slider("Total Sulfur Dioxide",min_value=mini[6],max_value=maxi[6],step=0.1)
        density = st.slider("Density",min_value=mini[7],max_value=maxi[7],step=0.0001)
        pH = st.slider("pH",min_value=mini[8],max_value=maxi[8],step=0.01)
        sulphates = st.slider("Sulphates",min_value=mini[9],max_value=maxi[9],step=0.01)
        alcohol = st.slider("Citric Acid",min_value=mini[10],max_value=maxi[10],step=0.1)

        features = [
            {
                "data id": "string",
                "fixed acidity": fixed_acidity,
                "volatile acidity": volatile_acidity,
                "citric acid": citric_acid,
                "residual sugar": residual_sugar,
                "chlorides": chlorides,
                "free sulfur dioxide": free_sulfur_dioxide,
                "total sulfur dioxide": total_sulfur_dioxide,
                "density": density,
                "pH": pH,
                "sulphates": sulphates,
                "alcohol": alcohol
            }
            ]
        
        
        submitted = st.form_submit_button("Submit")

        if submitted:
            data = requests.post(url=concat(request_url, "/predict"), json=features).json()
            if data:
                st.metric(label="Wine Quality",value=data[0]["value"])
            else:
                st.error("Error")


if __name__ == '__main__':
    main()

# %%



