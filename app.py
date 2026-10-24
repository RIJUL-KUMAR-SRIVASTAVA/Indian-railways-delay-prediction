
import streamlit as st
import pandas as pd
import joblib

# Load trained model and feature columns
model = joblib.load("railway_delay_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Indian Railways Train Delay Prediction")
st.write("Predict arrival delay at the next station.")

train_number = st.number_input(
    "Train Number", min_value=1, step=1
)

station_code = st.text_input(
    "Current Station Code", value="HWH"
).strip().upper()

seq = st.number_input(
    "Current Station Sequence Number",
    min_value=1,
    step=1
)

arrival_delay = st.number_input(
    "Current Arrival Delay (minutes)",
    value=0,
    step=1
)

if st.button("Predict Next Station Delay"):
    input_df = pd.DataFrame([{
        "train_number": train_number,
        "station_code": station_code,
        "seq": seq,
        "arrival_delay_min": arrival_delay
    }])

    input_df = pd.get_dummies(
        input_df,
        columns=["train_number", "station_code"]
    )

    input_df = input_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    prediction = model.predict(input_df)[0]

    st.success(
        f"Predicted next-station arrival delay: "
        f"{prediction:.2f} minutes"
    )

    if prediction > 0:
        st.write("The model predicts an arrival delay.")
    elif prediction < 0:
        st.write("The model predicts an early arrival.")
    else:
        st.write("The model predicts an on-time arrival.")
