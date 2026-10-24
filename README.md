
# Indian Railways Train Delay Prediction

## Project Overview
This project uses Machine Learning to predict arrival delay at the next railway station based on current station information.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning Models
- Linear Regression
- Decision Tree Regressor

The models were compared using Mean Absolute Error (MAE) and R² score. The Decision Tree was selected for the Streamlit application based on the initial evaluation.

## Features
- Enter train number
- Enter current station code
- Enter station sequence number
- Enter current arrival delay
- Predict the next station's arrival delay

## Run Locally
1. Install Python.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Place `railway_delay_model.pkl` and `model_columns.pkl` in the project folder.
4. Start the app:

   `streamlit run app.py`

## Limitations
This is a prototype trained on historical data. Predictions are estimates and are not live railway status information.
