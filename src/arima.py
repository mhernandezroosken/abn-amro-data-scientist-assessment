# general imports
from typing import List, Text, Union, Optional

# conda packages
import numpy as np
import pandas as pd
from attr import define, field
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon


# local imports


def make_input(df, client_nr, regressors):
    """
    Make input data for ARIMA
    """

    if client_nr is None:
        clients = np.sort(np.unique(df.client_nr))
    else:
        clients = np.array(client_nr).reshape((-1))

    df = df[df.client_nr.isin(clients)].groupby(['yearmonth_dt']).sum()
    df.index = pd.PeriodIndex(
        df.index,
        freq="M"
    )

    month = df.month
    y = df.has_credit_applications
    x = df[regressors]

    return month, x, y


class Arima1d:
    """
    Class for training and applying AutoArima from sktime
    """
    forecaster = None

    def __init__(self,
                 client_nr,
                 regressors: List[Text]
                 ):
        self.client_nr = client_nr
        self.regressors = regressors

    def fit(self,
            df):
        months, x_train, y_train = make_input(df, self.client_nr, self.regressors)
        self.forecaster = AutoARIMA(
            # sp=12,
            seasonal=False,
            suppress_warnings=True
        )

        self.forecaster.fit(
            y_train,
            X=x_train
        )

    def predict(self,
                df):
        moths, x_val, y_val = make_input(df, self.client_nr, self.regressors)
        y_pred = self.forecaster.predict(
            fh=ForecastingHorizon(moths.index, is_relative=False),
            X=x_val
            # n_prediods=6
        )
        return y_pred, y_val
