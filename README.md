# abn-amro-data-science-assessment-sme
ABN AMRO Data Science assessment. EDA and forecasting of SME mortgage data.

## Requirements

Install the requirements by running:

`conda en create --file=mortgages-py38.yaml`
`conda activate mortgages-py38`

## Running

To train and predict with ARIMA:
`python train_arima.py`

Similarly, to trian and predict with LSTM:
`python train_lstm.py`

## Project stucture

`EDA.py`: Python notebook with exploratory data analysis
`train_arima.py`: Python script for training (and predicting) with ARIMA.
`train_lstm.py`: Python script for trianing (and predicting) with LSTM.

