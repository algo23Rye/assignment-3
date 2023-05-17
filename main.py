import pandas as pd
from evalutaion import Evaluation
from prediction import *
import get_data

path = "./data/"
outcome_path = r"./outcome/"
code_list = [ 'SRM', 'CFM', 'cm', 'am', 'WHM']
# sample data
begin_date = '20150101'
end_date = '20230428'


def main():
    # 1. get main contract data and select sugar cotton corn soybeans wheat main future contract
    datahandler = get_data.Get_data(begin_date = begin_date, end_date = end_date)
    all_data = datahandler.get_all_main_future_data()
    all_data.to_csv(path + "main_contract.csv", encoding = 'gbk')
    select_df = all_data[all_data['code'].isin(code_list)]
    select_df.to_csv(path + "/select_contract.csv", encoding = 'gbk')

    # 2. construct the indicators
    data = pd.read_csv(path + "select_contract.csv", index_col = 0, encoding = 'gbk', parse_dates = ['date'])
    data = data.reset_index(drop = True).set_index(['code', 'date'])
    indicators = Technical_Indicator_Tablib.get_all_indicators(data)
    indicators.to_csv(path + "indicators.csv")

    # get standardized indicators
    indicators_s = Technical_Indicator_Tablib.get_standardized_indicators_by_zscore(indicators)
    indicators_s.to_csv(path + "indicators_standardized.csv")

    # get normalized indicators
    indicators_n = Technical_Indicator_Tablib.get_normalization_indicators(indicators)
    indicators_n.to_csv(path + "indicators_normalized.csv")

    # 3. train one by one and input_size is decided by the Pearsonr
    params = hyperparams()
    params.n1 = 1  # 1 day ahead
    # params.epochs, params.learning_rate = [500, 0.001] # for cm and srm
    params.epochs, params.learning_rate = [800, 0.001]  # for 'CFM', 'am', 'WHM'
    params.device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu")]
    params.seq_len = 22
    params.hidden_size1 = 128
    params.dropout_rate = 0.2
    params.hidden_size2 = 256

    # if use Mylstm1, add the following parameters
    # params.hidden_size3 = 256
    # params.hidden_size4 = 256
    # params.hidden_size5 = 128
    # params.hidden_size6 = 128

    features = pd.read_csv(path + "indicators_standardized.csv", index_col = [0, 1], parse_dates = True)
    data_price = pd.read_csv(path + "select_contract.csv", index_col = ['code', 'date'], parse_dates = True,
                             encoding = 'gbk')

    pre = LSTM_Prediction(features, data_price)

    for future_name in code_list:
        pre.get_return_as_objective(params.n1, 'close')
        features, obj = pre.feature_select(future_name)
        params.input_size = len(features.columns)
        outcome = LSTM_Prediction.rolling_pred(params, features, obj, future_name)
        outcome.to_csv(outcome_path + "prediction for " + future_name + ".rolling_pre.csv")

    # evaluation
    error_all = pd.DataFrame()
    for smybol in code_list:
        outcome = pd.read_csv(outcome_path + "/prediction for " + smybol + ".rolling_pre.csv", index_col = [0, 1],
                              parse_dates = [1])
        Evaluation.plot_nav(outcome)
        error = Evaluation.get_square_error(outcome)
        error_all = pd.concat([error_all, error])
    error_all.to_csv(outcome_path+"mean_square_error.csv")


if __name__ == "__main__":
    main()
