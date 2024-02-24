#!/usr/bin/env python
# coding: utf-8
"""
Train an (S)ARIMA(X) forecasting model for mortgages, and report its performance
"""

# general imports

# conda packages
import warnings

import numpy as np

# local imports
import pandas as pd

from src.arima import Arima1d
from src.metrics import get_metrics, print_metrics, create_prediction_csv
from src.load_data import get_data


def train_arima():
    """
    Main training function for the ARIMA model.
    We fit the Auto ARIMA model on the full train + val set.
    Then, we make forecasts on the test sets, output these to a csv and print metrics.
    """

    # constants
    nr_train = 22
    nr_val = 5
    nr_test = 5

    # cutoff probability for classification
    cutoff_prob = 0.15


    # load the dataset
    data = get_data(nr_train, nr_val, nr_test)
    df_train, df_val, df_tv, df_test, df_total, \
        complete_clients, incomplete_clients, regressors = data

    month_nr_to_yearmonth = {
        month_nr: ym for month_nr, ym in zip(
            np.sort(np.unique(df_total.yearmonth_nr)),
            np.sort(np.unique(df_total.yearmonth_str))
        )
    }

    # train ARIMA for each of the sequences
    y_pred_total, y_val_total = [], []
    client_nrs = complete_clients
    for client_nr in client_nrs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # fit  arima
            arima = Arima1d(client_nr, regressors)
            arima.fit(df_tv)

            # predict with arima.
            y_pred, y_val = arima.predict(df_test,)
            y_pred_total.append(y_pred)
            y_val_total.append(y_val)

    # predictions for all months for all clients
    y_pred_total = np.vstack(y_pred_total)
    y_val_total = np.vstack(y_val_total)

    metrics = get_metrics(y_val_total, y_pred_total, cutoff_prob=cutoff_prob)
    print_metrics(metrics)

    create_prediction_csv(
        y_val_total,
        y_pred_total,
        client_nrs,
        month_nr_to_yearmonth,
        nr_train,
        nr_val,
        nr_test,
        method_name='arima'
    )


if __name__ == '__main__':
    train_arima()