
import streamlit as st
import pandas as pd
import joblib

# Load saved model and training columns
model = joblib.load("railway_delay_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Load train routes
routes = pd.read_csv("train_routes.csv")

# Standardize train numbers
routes["train_number"] = (
    routes["train_number"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(5)
)

# Clean station codes
routes["station_code"] = (
    routes["station_code"]
    .astype(str)
    .str.strip()
)

# App title
st.title("Indian Railways Train Delay Prediction")
st.write(
    "Predict the arrival delay at the next station "
    "using the current station's information."
)

# Select train
train_options = (
    routes[["train_number", "train_name"]]
    .drop_duplicates()
    .sort_values("train_number")
)

train_options["label"] = (
    train_options["train_number"]
    + " - "
    + train_options["train_name"].astype(str)
)

selected_train = st.selectbox(
    "Select Train",
    train_options["label"].tolist()
)

train_number = selected_train.split(" - ")[0]

# Get stations for the selected train
train_stations = (
    routes[routes["train_number"] == train_number]
    .sort_values("seq")
    .drop_duplicates(subset=["station_code"])
)

if train_stations.empty:
    st.error("No stations found for this train.")
    st.stop()

# Select current station
station_options = train_stations["station_code"].tolist()

selected_station = st.selectbox(
    "Select Current Station",
    station_options
)

# Get station sequence number
seq = int(
    train_stations.loc[
        train_stations["station_code"] == selected_station,
        "seq"
    ].iloc[0]
)

st.write(f"Station sequence number: {seq}")

# Enter current arrival delay
arrival_delay = st.number_input(
    "Current Arrival Delay (minutes)",
    min_value=-500,
    max_value=2000,
    value=0,
    step=1
)

# Predict next station delay
if st.button("Predict Next Station Delay"):

    # Create input data
    input_df = pd.DataFrame([{
        "train_number": train_number,
        "station_code": selected_station,
        "seq": seq,
        "arrival_delay_min": arrival_delay
    }])

    # One-hot encode categorical features
    input_df = pd.get_dummies(
        input_df,
        columns=["train_number", "station_code"]
    )

    # Match training columns and their order
    input_df = input_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Generate prediction
    prediction = float(model.predict(input_df)[0])

    # Display result
    st.subheader("Prediction Result")

    st.metric(
        label="Predicted next-station arrival delay",
        value=f"{prediction:.2f} minutes"
    )

    if prediction > 0:
        st.warning(
            f"The model predicts approximately "
            f"{prediction:.2f} minutes of delay."
        )
    elif prediction < 0:
        st.info(
            f"The model predicts arrival approximately "
            f"{abs(prediction):.2f} minutes ahead of schedule."
        )
    else:
        st.success(
            "The model predicts arrival on schedule."
        )

st.caption(
    "This is an ML-based estimate, not a live railway status. "
    "Actual delay may differ from the prediction."
)
