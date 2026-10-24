# Indian Railways Train Delay Prediction

## About the Project

I built this project to predict how late a train might reach its next station. The prediction is based on the train number, current station, station sequence number, and the current arrival delay.

I used Python to prepare the data and train the model, then built a Streamlit app so users can try out predictions without running the model manually.

## Tools and Technologies

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## How It Works

1. Select a train from the dropdown list.
2. Choose the current station.
3. Enter the train's current arrival delay in minutes.
4. Click **Predict Next Station Delay** to get the estimated delay at the next station.

The app uses a trained Decision Tree Regressor to make the prediction.

## Model Performance

I evaluated the model on a test set of 401 records.

- **Mean Absolute Error (MAE):** 28.72 minutes
- **R² Score:** 0.9151

These are results from a random train-test split of a 3,000-record sample. The actual delay can be different from the predicted delay, and performance on unseen train journeys may vary.

## Run the Project Locally

1. Install Python.
2. Download or clone this repository.
3. Open a terminal in the project folder.
4. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Make sure these files are present in the project folder:
   - `railway_delay_model.pkl`
   - `model_columns.pkl`
   - `train_routes.csv`

6. Start the app:

   ```bash
   streamlit run app.py
   ```

## Limitations

This project gives an ML-based estimate using the available dataset. It does not fetch live train running status from Indian Railways, so the predicted delay may not match the actual arrival time.