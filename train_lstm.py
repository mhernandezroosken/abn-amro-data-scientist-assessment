"""
Train an LSTM forecasting model for mortgages, and report its performance
"""

# general imports

# conda package imports
import numpy as np
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet

# local imports
from src.metrics import get_metrics, print_metrics, create_prediction_csv
from src.load_data import get_data
from src.lstm import LSTMModel, LightningModel


def get_data_loaders(df_train, df_val, df_tv, df_test, df_total, nr_train, nr_val, nr_test, regressors, target):
    encoder_length = nr_train - nr_val - 4

    # train dataset
    train_dataset = TimeSeriesDataSet(
        df_train,
        time_idx='yearmonth_nr',
        target=target,
        group_ids=['client_nr_cat'],
        max_encoder_length=encoder_length,
        max_prediction_length=nr_val,
        time_varying_unknown_reals=regressors + [target],
        time_varying_known_categoricals=['month_cat'],
        static_categoricals=['client_nr_cat'],
    )

    # validation dataset
    val_dataset = TimeSeriesDataSet(
        df_tv,
        time_idx='yearmonth_nr',
        target=target,
        group_ids=['client_nr_cat'],
        max_encoder_length=encoder_length,
        max_prediction_length=nr_val,
        time_varying_unknown_reals=regressors + [target],
        time_varying_known_categoricals=['month_cat'],
        static_categoricals=['client_nr_cat'],
        categorical_encoders=train_dataset.categorical_encoders,
        scalers=train_dataset.scalers,
        target_normalizer=train_dataset.target_normalizer,
        min_prediction_idx=np.min(df_val.yearmonth_nr)  # predict only the val months
    )

    # test dataset
    test_dataset = TimeSeriesDataSet(
        df_total,
        time_idx='yearmonth_nr',
        target=target,
        group_ids=['client_nr_cat'],
        max_encoder_length=encoder_length,
        max_prediction_length=nr_val,
        time_varying_unknown_reals=regressors + [target],
        time_varying_known_categoricals=['month_cat'],
        static_categoricals=['client_nr_cat'],
        categorical_encoders=train_dataset.categorical_encoders,
        scalers=train_dataset.scalers,
        target_normalizer=train_dataset.target_normalizer,
        min_prediction_idx=np.min(df_test.yearmonth_nr)  # predict only the test months
    )

    train_dl = train_dataset.to_dataloader(
        batch_size=256,
        shuffle=True
    )

    val_dl = val_dataset.to_dataloader(
        batch_size=val_dataset.__len__(),
        shuffle=False
    )

    test_dl = test_dataset.to_dataloader(
        batch_size=test_dataset.__len__(),
        shuffle=False
    )

    return train_dataset, val_dataset, test_dataset, train_dl, val_dl, test_dl, encoder_length


def train_lstm():
    # train/val/test proportions in months
    nr_train = 24
    nr_val = 4
    nr_test = 4
    # use cuda for training
    device = 'cuda'
    # training target
    target = 'has_credit_applications'

    # load the csv's into dataframes
    df_train, df_val, df_tv, df_test, df_total, \
        complete_clients, incomplete_clients, \
        regressors = get_data(nr_train, nr_val, nr_test)

    # map between month numbers and yearmonths
    month_nr_to_yearmonth = {
        month_nr: ym for month_nr, ym in zip(
            np.sort(np.unique(df_total.yearmonth_nr)),
            np.sort(np.unique(df_total.yearmonth_str))
        )
    }

    # get datasets + dataloaders from pytorch_forecasting
    train_dataset, val_dataset, test_dataset, train_dl, val_dl, test_dl, encoder_length = get_data_loaders(
        df_train,
        df_val,
        df_tv,
        df_test,
        df_total,
        nr_train,
        nr_val,
        nr_test,
        regressors,
        target
    )

    nr_months = 12
    nr_clients = max(train_dataset.categorical_encoders['client_nr_cat'].classes_.values()) + 1

    model = LSTMModel(
        input_channels=7,
        input_length=encoder_length,
        forecast_channels=1,
        forecast_length=nr_val,
        nr_hidden_channels=128,
        nr_layers=2,
        nr_months=nr_months,
        nr_clients=nr_clients,
        dropout=0.3
    )

    pl_model = LightningModel(
        model,
        lr=5e-4,
        weight_decay=5e-5,
        p_cutoff=0.15
    )

    trainer = pl.Trainer(
        gpus=1,
        precision=32,
        max_epochs=150,
    )

    # train the model
    trainer.fit(pl_model, train_dl, val_dl)

    # make predictions on the test set
    items, (y, _) = next(iter(test_dl))
    x_train = items['encoder_cont'].to(device)
    y_val = items['decoder_target'].numpy()

    # carry out evaluation on cuda
    y_pred_logits, y_pred_probs = trainer.predict(dataloaders=test_dl)[0]
    y_pred_probs = y_pred_probs.numpy()

    metrics = get_metrics(y_val, y_pred_probs, cutoff_prob=0.1)
    print_metrics(metrics)

    create_prediction_csv(
        y_val,
        y_pred_probs,
        complete_clients,
        month_nr_to_yearmonth,
        nr_train,
        nr_val,
        nr_test,
        method_name='lstm'
    )


if __name__ == '__main__':
    train_lstm()
