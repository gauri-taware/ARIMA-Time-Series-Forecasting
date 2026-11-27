# ARIMA Time Series Forecasting

## Project Overview
This project implements time series forecasting using the ARIMA (AutoRegressive Integrated Moving Average) model in Python.

It uses historical numerical data to train an ARIMA model, predict future values, evaluate forecasting performance, and visualize actual and predicted values.

## Technologies Used
- Python 3.10
- Pandas
- NumPy
- pmdarima
- Matplotlib
- Scikit-learn

## Project Structure
```text
ARIMA-Time-Series-Forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── data.csv
├── arima_model.py
└── arima_forecast.png
```

## Features
- Loads time-series data from a CSV file.
- Converts date values into datetime format.
- Splits data into training and testing datasets.
- Uses Auto ARIMA to select suitable model parameters.
- Generates future forecasts.
- Calculates MAE and RMSE.
- Visualizes training, actual, and predicted values.

## Installation

Clone the repository:

```bash
git clone https://github.com/gauri-taware/ARIMA-Time-Series-Forecasting
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Execution

Run the following command:

```bash
python arima_model.py
```

## Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## Output
The program generates:
- Selected ARIMA model order.
- Forecasting performance metrics.
- Six future forecast values.
- A visualization saved as `arima_forecast.png`.

## Dataset
The project uses monthly time-series observations stored in `data.csv`.

## Author
Gauri R. Taware
