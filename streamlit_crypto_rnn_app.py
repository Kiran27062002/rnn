
# ================================
# 📊 Streamlit Crypto RNN App
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Crypto RNN Predictor", layout="wide")

st.title("📈 Crypto Currency Prediction Dashboard")

# Sidebar options
st.sidebar.header("⚙️ Settings")
data_dir = st.sidebar.text_input("Dataset Folder", "time-series-top-100-crypto-currency-dataset")
pred_dir = st.sidebar.text_input("Prediction Folder", "predictions_per_coin")

coins = [f.replace('.csv','') for f in os.listdir(data_dir) if f.endswith('.csv')]
selected_coin = st.sidebar.selectbox("Select Coin", coins)

if selected_coin:
    data_path = os.path.join(data_dir, f"{selected_coin}.csv")
    df = pd.read_csv(data_path)
    st.subheader(f"Raw Data for {selected_coin}")
    st.dataframe(df.tail())

    pred_path = os.path.join(pred_dir, f"{selected_coin}_pred.csv")
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        st.subheader("📊 Actual vs Predicted Close Prices")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(pred_df['actual'], label='Actual')
        ax.plot(pred_df['predicted'], label='Predicted')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning(f"No prediction file found for {selected_coin}.")
