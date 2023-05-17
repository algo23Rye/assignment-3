import pandas as pd
import torch
from torch import nn
import numpy as np
from indicator import Technical_Indicator_Tablib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class SelectItem(nn.Module):
    '''
    used for getting the hidden layers output of lstm
    '''

    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index][0]


class hyperparams():
    '''
    use for storing the hyperparams of the model
    '''
    pass


class Mylstm1(nn.Module):
    '''
    This model is constructed according to the paper.
    '''

    def __init__(self, input_size, hidden_size1, dropout_rate, hidden_size2, hidden_size3, hidden_size4,
                 hidden_size5, hidden_size6):
        super(Mylstm1, self).__init__()
        self.sequence = nn.Sequential(
            nn.LSTM(input_size = input_size, hidden_size = hidden_size1, num_layers = 1),
            SelectItem(1),
            nn.Dropout(p = dropout_rate),
            nn.LSTM(input_size = hidden_size1, hidden_size = hidden_size2, num_layers = 1),
            SelectItem(1),
            nn.Dropout(p = dropout_rate),
            nn.LSTM(input_size = hidden_size2, hidden_size = hidden_size3, num_layers = 1),
            SelectItem(1),
            nn.Linear(in_features = hidden_size3, out_features = hidden_size4),
            nn.Dropout(p = dropout_rate),
            nn.LSTM(input_size = hidden_size4, hidden_size = hidden_size5, num_layers = 1),
            SelectItem(1),
            nn.Linear(in_features = hidden_size5, out_features = hidden_size6),
            nn.Dropout(p = dropout_rate),
            nn.Linear(in_features = hidden_size6, out_features = 1),
            nn.ReLU())  # if use price as objective

    def forward(self, features):
        y = self.sequence(features)
        if len(y.shape) == 3:
            pred = y[-1, :, :]
        elif len(y.shape) == 2:
            pred = y[-1, :]
        return pred


class Mylstm2(nn.Module):
    '''
    This model is simpler than Mylstm1 with less layers.
    '''

    def __init__(self, input_size, hidden_size1, dropout_rate, hidden_size2):
        super(Mylstm2, self).__init__()
        self.sequence = nn.Sequential(
            nn.LSTM(input_size = input_size, hidden_size = hidden_size1, num_layers = 1),
            SelectItem(1),
            nn.Dropout(p = dropout_rate),
            nn.Linear(in_features = hidden_size1, out_features = hidden_size2),
            nn.Dropout(p = dropout_rate),
            nn.Linear(in_features = hidden_size2, out_features = 1))
        # nn.ReLU()) #if use price as objective

    def forward(self, features):
        y = self.sequence(features)
        if len(y.shape) == 3:
            pred = y[-1, :, :]
        elif len(y.shape) == 2:
            pred = y[-1, :]
        return pred


class LSTM_Prediction:
    def __init__(self, features, data_price, path = "./"):
        self.features = features
        self.data_price = data_price
        self.path = path

    @staticmethod
    def data_processing(features, objective, seq_len):
        '''

        use to change the dimension of data ->three dimensions as the input of lstm

        :param features: lag features
        :param objective:
        :param seq_len: length of time series
        :return: three dimensions data
        '''

        dataX, dataY, = [], []
        for i in range(len(features) - seq_len):
            dataX.append(torch.as_tensor(features.iloc[i:i + seq_len, :].values))
            datax = torch.stack(dataX, dim = 1).float()
            dataY.append(torch.as_tensor(np.array(objective.iloc[i + seq_len - 1, -1])))
            datay = torch.stack(dataY, dim = 0).float()
            datay = datay.view(datay.shape[0], -1)

        return datax, datay

    def get_price_as_objective(self, n1, price_type = 'close', dmax = 1, dmin = 0):
        '''
        :param n1: n1 day ahead price as the objective, to shift the features
        :param price_type: price type as the objective
        :return: the shifting n1 features, to match the index of price, in order to train
        '''

        self.features_lag = self.features.groupby('code').apply(lambda x: x.shift(n1)).dropna()

        price = self.data_price.loc[:, [price_type]].groupby('code').apply(
            lambda x: (x - x.expanding().min()) * (dmax - dmin) / (x.expanding().max() - x.expanding().min()) + dmin)

        price.dropna(inplace = True)
        # drop the first 250 days data
        n = 250
        price = price.groupby('code').apply(lambda x: x.iloc[n:, :]).droplevel(level = 0)
        self.objective_all = price.loc[self.features_lag.index, [price_type]]
        return self.objective_all

    def get_return_as_objective(self, n1, price_type = 'close'):
        '''
        :param n1: n1 day ahead return as the objective, to shift the features
        :param price_type: price type as the objective
        :return: the shifting n1 features, to match the index of price, in order to train
        '''

        self.features_lag = self.features.groupby('code').apply(lambda x: x.shift(n1)).dropna()

        r = self.data_price.loc[:, [price_type]].groupby('code').apply(
            lambda x: (x[price_type] / x[price_type].shift(1) - 1) * 100).to_frame().droplevel(0)
        r.columns = [str(n1) + '_ahead_return']

        r.dropna(inplace = True)

        self.objective_all = r.loc[self.features_lag.index, [str(n1) + '_ahead_return']]
        return self.objective_all

    def feature_select(self, future_name):
        '''
        To select the features for one future by pearsonr
        :param future_name:
        '''

        select = Technical_Indicator_Tablib.indicators_selected_by_Pearsonr(self.data_price).loc[future_name, :][0]
        # select=pd.read_csv(self.path+"select_indicators.csv",index_col = 0).loc[future_name,][0]
        select_features = self.features_lag.loc[future_name, select]
        obj = self.objective_all.loc[future_name, :]
        return select_features, obj

    @staticmethod
    def split_pred(params, features, obj, future_name, ratio: float = 0.8):
        '''
        used to split the whole data into two parts, one for training and one for testing.

        :param params: lstm parameters
        :param features: used as input features in the model
        :param obj: objective
        :param future_name:
        :param ratio: train sample length/ test sample length
        :return:
        '''
        datax, datay = LSTM_Prediction.data_processing(features, obj, params.seq_len)
        start_train_end = int(ratio * datax.shape[1])

        # use the test set index
        outcome = pd.DataFrame(index = pd.MultiIndex.from_product(
            [[future_name], list(features.iloc[params.seq_len + start_train_end:, :].index)], names = ['code', 'date']),
            columns = ['y_pred_' + str(params.n1) + 'ahead', 'y_true_' + str(params.n1) + 'ahead'])

        # for start_train_end in range(start_train_end, datax.shape[1] - 1):
        train_x = datax[:, :start_train_end, :].to(params.device)
        train_y = datay[:start_train_end, :].to(params.device)
        test_x = datax[:, start_train_end:, :].to(params.device)
        test_y = datay[start_train_end:, :].to(params.device)

        # model = Mylstm1(params.input_size, params.hidden_size1, params.dropout_rate, params.hidden_size2,
        #                 params.hidden_size3, params.hidden_size4, params.hidden_size5, params.hidden_size6).to(
        #     params.device)

        model = Mylstm2(params.input_size, params.hidden_size1, params.dropout_rate, params.hidden_size2, ).to(
            params.device)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr = params.learning_rate)

        for epoch in range(params.epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(train_x)
            loss = criterion(pred, train_y)
            loss.backward()
            optimizer.step()
        print(loss)

        model.eval()
        pred_test_y = model(test_x)
        outcome.iloc[:, 0] = pred_test_y.detach().cpu().numpy().squeeze()
        outcome.iloc[:, 1] = test_y.detach().cpu().numpy().squeeze()

        return outcome

    @staticmethod
    def expanding_pred(params, features, obj, future_name, ratio: float = 0.8):
        '''
        expanding windows to predict one step forward
        '''

        datax, datay = LSTM_Prediction.data_processing(features, obj, params.seq_len)
        start_train_end = int(ratio * datax.shape[1])
        # use the test set index
        outcome = pd.DataFrame(index = pd.MultiIndex.from_product(
            [[future_name], list(features.iloc[params.seq_len + start_train_end:, :].index)], names = ['code', 'date']),
            columns = ['y_pred_' + str(params.n1) + 'ahead', 'y_true_' + str(params.n1) + 'ahead'])

        for i in range(start_train_end, start_train_end + 20):
            # for i in range(start_train_end, datax.shape[1]):
            train_x = datax[:, :i, :].to(params.device)
            train_y = datay[:i, :].to(params.device)
            test_x = datax[:, :i + 1, :].to(params.device)
            test_y = datay[:i + 1, :].to(params.device)

            model = Mylstm1(params.input_size, params.hidden_size1, params.dropout_rate, params.hidden_size2,
                            params.hidden_size3, params.hidden_size4, params.hidden_size5, params.hidden_size6).to(
                params.device)

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr = params.learning_rate)

            for epoch in range(params.epochs):
                model.train()
                optimizer.zero_grad()
                pred = model(train_x)
                loss = criterion(pred, train_y)
                loss.backward()
                optimizer.step()
            print(loss)

            model.eval()
            pred_test_y = model(test_x)
            outcome.iloc[i - start_train_end, 0] = pred_test_y.detach().cpu().numpy().squeeze()[-1]
            outcome.iloc[i - start_train_end, 1] = test_y.detach().cpu().numpy().squeeze()[-1]

        return outcome

    @staticmethod
    def rolling_pred(params, features, obj, future_name, ratio = 0.2):
        '''
        rolling window of prediction.
        '''

        datax, datay = LSTM_Prediction.data_processing(features, obj, params.seq_len)
        start_train_end = int(ratio * datax.shape[1])
        # use the test set index
        outcome = pd.DataFrame(index = pd.MultiIndex.from_product(
            [[future_name], list(features.iloc[params.seq_len + start_train_end:, :].index)], names = ['code', 'date']),
            columns = ['y_pred_' + str(params.n1) + 'ahead', 'y_true_' + str(params.n1) + 'ahead'])

        # for i in range(start_train_end, len(datax)-1):
        for i in range(start_train_end, datax.shape[1]):
            train_x = datax[:, i - start_train_end:i, :].to(params.device)
            train_y = datay[i - start_train_end:i, :].to(params.device)
            test_x = datax[:, i + 1 - start_train_end: i + 1, :].to(params.device)
            test_y = datay[i + 1 - start_train_end:i + 1, :].to(params.device)

            # model = Mylstm1(params.input_size, params.hidden_size1, params.dropout_rate, params.hidden_size2,
            #                 params.hidden_size3, params.hidden_size4, params.hidden_size5, params.hidden_size6).to(
            #     params.device)

            model = Mylstm2(params.input_size, params.hidden_size1, params.dropout_rate, params.hidden_size2, ).to(
                params.device)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr = params.learning_rate)

            for epoch in range(params.epochs):
                model.train()
                optimizer.zero_grad()
                pred = model(train_x)
                loss = criterion(pred, train_y)
                loss.backward()
                optimizer.step()
            print(loss)
            model.eval()
            pred_test_y = model(test_x)
            outcome.iloc[i - start_train_end, 0] = pred_test_y.detach().cpu().numpy().squeeze()[-1]
            outcome.iloc[i - start_train_end, 1] = test_y.detach().cpu().numpy().squeeze()[-1]
            # outcome.to_csv("prediction for " + future_name + ".csv")
        return outcome
