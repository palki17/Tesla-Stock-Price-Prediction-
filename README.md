# 🚗 Tesla Stock Price Prediction

## Project Overview
This project predicts Tesla (TSLA) stock prices using Deep Learning models — SimpleRNN and LSTM — trained on historical data from 2010 to 2020.

## 📁 Project Structure
```
Tesla_Project/
    ├── app.py                      # Streamlit web app
    ├── TSLA.csv                    # Dataset
    ├── tsla_simplernn_model.keras  # Trained SimpleRNN model
    ├── tsla_lstm_model.keras       # Trained LSTM model
    ├── tsla_tuned_lstm_model.keras # Tuned LSTM model
    ├── tsla_scaler.joblib          # MinMaxScaler
    ├── requirements.txt            # Dependencies
    └── README.md                   # This file
```

## 🧠 Models Used
| Model | Description |
|---|---|
| SimpleRNN | Basic recurrent neural network |
| LSTM | Long Short-Term Memory network |
| Tuned LSTM | LSTM with best hyperparameters from manual grid search |

## 📊 Features
- Historical price visualization with moving averages
- 1-day, 5-day, and 10-day future price forecasting
- Interactive model selection and forecast slider
- SimpleRNN vs LSTM performance comparison
- Raw data viewer with year filter

## 🚀 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the app
```bash
streamlit run app.py
```

## 📈 Dataset
- Source: TSLA.csv
- Date Range: June 2010 – February 2020
- Records: 2,416 trading days
- Target: Adjusted Closing Price

## 🔧 Hyperparameters Tuned
- Units: [32, 64]
- Dropout Rate: [0.2, 0.3]
- Learning Rate: [0.001, 0.0005]

## 📉 Evaluation Metrics
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

## 🏆 Conclusion
LSTM outperforms SimpleRNN on Tesla stock data because Tesla had a massive multi-month bull run in 2019–2020. LSTM's gating mechanism captures these long-term dependencies better than SimpleRNN which suffers from the vanishing gradient problem.

## 👩‍💻 Tech Stack
- Python 3.10+
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib
- Streamlit
