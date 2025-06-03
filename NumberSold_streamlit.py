import streamlit as st
import joblib
import numpy as np

model = joblib.load("Number Sold.pkl")

def main():
    st.title("Machine Learning Model Deployment")

    store = st.slider("store", min_value = 0, max_value = 6, value = 0)
    product = st.slider("product", min_value = 0, max_value = 9, value = 0)
    year = st.slider("year", min_value = 2010, max_value = 2025, value = 2010)
    month = st.slider("month", min_value = 1, max_value = 12, value = 1)
    day = st.slider("day", min_value = 1, max_value = 31, value = 1)
    
    if st.button("Make Prediction"):
        features = [store, product, year, month, day]
        result = make_prediction(features)
        st.success(f"The prediction is: {result}")

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == "__main__":
    main()