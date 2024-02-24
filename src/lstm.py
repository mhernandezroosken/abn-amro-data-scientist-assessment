import torch
from torch import nn
import pytorch_lightning as pl

from src.metrics import get_metrics


class LightningModel(pl.LightningModule):
    """
    Model wrapper class for Pytorch Lightning.
    This model is able to define the optimizer, the train step and the validation step.
    Please note that the combination of sigmoid and BCELoss can the tendency to not converge
    (exploding gradient problem).
    """
    def __init__(self,
                 model: nn.Module,
                 lr: float,
                 weight_decay: float,
                 p_cutoff: float
                 ):
        super().__init__()
        self.p_cutoff = p_cutoff
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        # NOTE: USE THIS INSTEAD OF BCELoss TO AVOID NUMERICAL INSTABILITY AND FAILURE TO CONVERGE
        self.loss = torch.nn.BCELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        items, (y, _) = train_batch
        # x_train = items['encoder_cont'].to(self.device)
        y_target = items['decoder_target'].to(self.device)

        y_pred_logits, y_pred_probs = self.forward(train_batch)

        loss = self.loss(y_pred_probs, y_target)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        items, (y, _) = val_batch
        # x_train = items['encoder_cont'].to(self.device)
        y_val = items['decoder_target'].to(self.device)

        y_pred_logits, y_pred_probs = self.forward(val_batch)

        val_loss = self.loss(y_pred_probs, y_val)
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return y_val, y_pred_probs

    def validation_epoch_end(self, validation_step_outputs):
        y_val, y_pred = validation_step_outputs[0]
        metrics = get_metrics(
            y_val.detach().cpu().numpy(),
            y_pred.detach().cpu().numpy(),
            cutoff_prob=self.p_cutoff
        )
        for key, value in metrics.items():
            if 'conf_mat' in key:
                # don't log the confusion matrix, since np.ndarray is not supported
                continue
            self.log('val_' + key, value, prog_bar=True, on_step=False, on_epoch=True)

        return metrics

    def forward(self, x):
        return self.model.forward(x)


class LSTMModel(nn.Module):
    def __init__(self, input_channels,
                 input_length,
                 forecast_channels,
                 forecast_length,
                 nr_hidden_channels,
                 nr_layers,
                 nr_months,
                 nr_clients,
                 dropout=0.10,
                 ):
        """
        LSTM model for mortgage forecasting.
        Create forecast probablities for a window in the future.
        Returns these both as logits and probabilities
        Args:
            input_channels: number of input channels
            input_length: time length of the input
            forecast_channels: number of forecast channels (1)
            forecast_length: length of the forecast channels (4)
            nr_hidden_channels: number of hidden channels in the LSTM
            nr_layers: number of layers in the LSTM stack
            nr_months: number of months present in the dataset (12)
            nr_clients: number of clients present in the dataset
            dropout: dropout percentage for the LSTM
        """
        super().__init__()

        self.input_channels = input_channels + nr_months
        self.input_length = input_length
        self.forecast_channels = forecast_channels
        self.forecast_length = forecast_length
        self.nr_hidden_channels = nr_hidden_channels
        self.nr_months = nr_months
        self.nr_clients = nr_clients
        self.nr_layers = nr_layers
        self.dropout = dropout

        self.forecast_size_flat = self.forecast_channels * self.forecast_length

        # layers = OrderedDict()
        in_channels = self.input_channels
        out_channels = self.nr_hidden_channels

        # for i in range(nr_layers):
        #     if i == nr_layers - 1:
        #         out_channels = self.forecast_size_flat
        self.lstm_stack = nn.LSTM(input_size=in_channels,
                                  hidden_size=self.nr_hidden_channels,
                                  dropout=self.dropout,
                                  num_layers=self.nr_layers,
                                  batch_first=True)
        # layers[f'lstms'] = lstm_layers
        self.linear = nn.Linear(self.nr_hidden_channels, self.forecast_size_flat)
        # self.lstm_stack = nn.Sequential(layers)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # if isinstance(x, tuple):
        # unpack the dataloader output
        items, (y, _) = x
        x = items['encoder_cont']
        x_cat = items['encoder_cat'][..., 1]
        x_cat = nn.functional.one_hot(x_cat, num_classes=self.nr_months)
        x = torch.cat((x, x_cat), dim=-1)

        # apply lstm stack
        x, (h_n, c_n) = self.lstm_stack(x)
        x = x[..., -1, :]

        # apply linear layer
        x = self.linear(x)

        # provide final output
        if self.forecast_channels > 1:
            shape = (x.shape[0], self.forecast_length, self.forecast_channels)
        else:
            shape = (x.shape[0], self.forecast_length)
        x = torch.reshape(x, shape)

        # apply activation
        x_probs = self.activation(x)
        x_logits = x

        return x_logits, x_probs
