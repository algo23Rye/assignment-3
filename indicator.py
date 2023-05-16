import pandas as pd
import talib
from scipy.stats import pearsonr
# from scipy import stats
import numpy as np


# use talib to calculate the indicators

class Technical_Indicator_Tablib:

    @staticmethod
    def get_two_ind(fun, *series):
        # used for getting two return values for groupby + apply

        o1, o2 = fun(*series)
        return pd.concat([o1, o2], axis = 1)

    @staticmethod
    def get_three_ind(fun, *series):
        # used for getting three return values for groupby + apply

        o1, o2, o3 = fun(*series)
        return pd.concat([o1, o2, o3], axis = 1)

    @staticmethod
    def standardized_indicators_zscores(indicators_for_one_future):
        '''
        :param indicators_for_one_future: dataframe with index ['code']['date']
        and just one type of codes: one future
        '''

        cummean = indicators_for_one_future.expanding().mean()
        cumstd = indicators_for_one_future.expanding().std()

        indicator_s = (indicators_for_one_future.sub(cummean)).div(cumstd)
        return indicator_s

    @staticmethod
    def get_cycle_indicators(data):
        '''data: a dataframe with index  ['code','date']'''

        close_data = data.loc[:, 'close']  # get close_series

        cycle_indicators_functions = [talib.HT_DCPERIOD, talib.HT_DCPHASE, talib.HT_PHASOR, talib.HT_SINE,
                                      talib.HT_TRENDMODE]
        cycle_indicators_name = ["HT DCPERIOD", "HT DCPHASE", "INPHASE", " QUADRATURE", "SINE", "LEADSINE",
                                 "HT TRENDMODE"]
        cycle_indicators = pd.DataFrame(index = data.index)

        for fun in cycle_indicators_functions:
            if (fun == talib.HT_PHASOR) or (fun == talib.HT_SINE):
                ind_series = close_data.groupby('code').apply(lambda x: Technical_Indicator_Tablib.get_two_ind(fun, x))
            else:
                ind_series = close_data.groupby('code').apply(lambda x: fun(x))

            cycle_indicators = pd.concat([cycle_indicators, ind_series], axis = 1)

        cycle_indicators.columns = cycle_indicators_name
        return cycle_indicators

    @staticmethod
    def get_momentum_indicators(data):

        momentum_indicators = pd.DataFrame(index = data.index)

        momentum_indicators_name = ['ADX', 'ADXR', 'APO', 'AROONDOWN', 'AROONUP', 'AROONOSC', 'BOP', 'CCI', 'CMO', 'DX',
                                    'MACD_dif', 'MACD_dem', 'MACD_histogram', 'MACDEXT_diff', 'MACDEXT_dem',
                                    'MACDEXT_histogram',
                                    'MACDFIX_diff', 'MACDFIX_dem', 'MACDFIXT_histogram', 'MFI', 'MINUS_DI', 'MINUS_DM',
                                    'MOM',
                                    'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'RSI', 'STOCH_slowk',
                                    'STOCH_slowd', 'STOCHF_fastk', 'STOCHF_fastd', 'STOCHRSI_fastk', 'STOCHRSI_fastd',
                                    'TRIX', 'ULTOSC', 'WILLR']

        momentum_indicators_functions = [talib.ADX, talib.ADXR, talib.APO, talib.AROON, talib.AROONOSC, talib.BOP,
                                         talib.CCI, talib.CMO, talib.DX, talib.MACD, talib.MACDEXT, talib.MACDFIX,
                                         talib.MFI, talib.MINUS_DI, talib.MINUS_DM, talib.MOM, talib.PLUS_DI,
                                         talib.PLUS_DM, talib.PPO, talib.ROC, talib.ROCP, talib.ROCR, talib.ROCR100,
                                         talib.RSI, talib.STOCH, talib.STOCHF, talib.STOCHRSI, talib.TRIX, talib.ULTOSC,
                                         talib.WILLR]

        # parameter are high, low, close

        h_l_c_list = [talib.ADX, talib.ADXR, talib.CCI, talib.DX, talib.MINUS_DI, talib.PLUS_DI, talib.STOCH,
                      talib.STOCHF, talib.ULTOSC, talib.WILLR]
        # parameter is close
        c_list = [talib.APO, talib.CMO, talib.MACD, talib.MACDEXT, talib.MACDFIX, talib.MOM, talib.PPO,
                  talib.ROC, talib.ROCP, talib.ROCR, talib.ROCR100, talib.RSI, talib.STOCHRSI, talib.TRIX, ]

        # parameter are high, low
        h_l_list = [talib.AROON, talib.AROONOSC, talib.MINUS_DM, talib.PLUS_DM, ]
        # parameter are open, high, low, close
        o_h_l_c_list = [talib.BOP, ]
        # parameter are high, low, close, volume
        # h_l_c_v_list = [talib.MFI, ]

        for fun in momentum_indicators_functions:
            if fun in c_list:
                if fun in [talib.MACD, talib.MACDEXT, talib.MACDFIX]:
                    ind_series = data.groupby('code').apply(
                        lambda x: Technical_Indicator_Tablib.get_three_ind(fun, x['close']))
                elif fun == talib.STOCHRSI:
                    ind_series = data.groupby('code').apply(
                        lambda x: Technical_Indicator_Tablib.get_two_ind(fun, x['close']))
                else:
                    ind_series = data.groupby('code').apply(lambda x: fun(x['close']))

            elif fun in h_l_list:
                if fun == talib.AROON:
                    ind_series = data.groupby('code').apply(
                        lambda x: Technical_Indicator_Tablib.get_two_ind(fun, x['high'], x['low']))
                else:
                    ind_series = data.groupby('code').apply(lambda x: fun(x['high'], x['low']))

            elif fun in h_l_c_list:
                if fun in [talib.STOCH, talib.STOCHF]:
                    ind_series = data.groupby('code').apply(
                        lambda x: Technical_Indicator_Tablib.get_two_ind(fun, x['high'], x['low'], x['close']))
                else:
                    ind_series = data.groupby('code').apply(lambda x: fun(x['high'], x['low'], x['close']))
            elif fun in o_h_l_c_list:
                ind_series = data.groupby('code').apply(
                    lambda x: fun(x['open'], x['high'], x['low'], x['close']))
            else:
                ind_series = data.groupby('code').apply(
                    lambda x: fun(x['high'], x['low'], x['close'], x['volume']))

            if len(ind_series.index[0]) == 3:
                ind_series = ind_series.droplevel(level = 0)

            momentum_indicators = pd.concat([momentum_indicators, ind_series], axis = 1)
        momentum_indicators.columns = momentum_indicators_name

        return momentum_indicators

    @staticmethod
    def get_volatility_indicators(data):
        volatility_indicators = pd.DataFrame(index = data.index)

        volatility_indicators_name = ['ATR', 'NATR', 'TRANGE']

        volatility_indicators_functions = [talib.ATR, talib.NATR, talib.TRANGE]
        for fun in volatility_indicators_functions:
            ind_series = data.groupby('code').apply(lambda x: fun(x['high'], x['low'], x['close']))
            if len(ind_series.index[0]) == 3:
                ind_series = ind_series.droplevel(level = 0)
            volatility_indicators = pd.concat([volatility_indicators, ind_series], axis = 1)

        volatility_indicators.columns = volatility_indicators_name
        return volatility_indicators

    @staticmethod
    def get_volume_indicators(data):
        volume_indicators = pd.DataFrame(index = data.index)

        volume_indicators_name = ['AD', 'ADOSC', 'OBV']

        volume_indicators_functions = [talib.AD, talib.ADOSC, talib.OBV]
        for fun in volume_indicators_functions:
            if fun == talib.OBV:
                ind_series = data.groupby('code').apply(lambda x: fun(x['close'], x['volume']))
            else:
                ind_series = data.groupby('code').apply(lambda x: fun(x['high'], x['low'], x['close'], x['volume']))

            if len(ind_series.index[0]) == 3:
                ind_series = ind_series.droplevel(level = 0)
            volume_indicators = pd.concat([volume_indicators, ind_series], axis = 1)

        volume_indicators.columns = volume_indicators_name
        return volume_indicators

    @staticmethod
    def get_all_indicators(data):
        cycle_indicators = Technical_Indicator_Tablib.get_cycle_indicators(data)
        momentum_indicators = Technical_Indicator_Tablib.get_momentum_indicators(data)
        volatility_indicators = Technical_Indicator_Tablib.get_volatility_indicators(data)
        volume_indicators = Technical_Indicator_Tablib.get_volume_indicators(data)
        indicator_list = [cycle_indicators, momentum_indicators, volatility_indicators, volume_indicators]
        indicators = pd.DataFrame(index = cycle_indicators.index)
        for ind in indicator_list:
            indicators = pd.concat([indicators, ind], axis = 1)

        indicators.dropna(inplace = True)
        return indicators

    @staticmethod
    def pearson_corr(x, y):
        '''
        :param x: series of one indicator of future name
        :param y: series of close prices of future name
        '''
        corr = np.corrcoef(x, y)[0, 1]
        return corr

    @staticmethod
    def get_corr(future_df, y):
        '''

        :param future_df: dataframe of all indicators of future name
        :param y: series of close prices of future name

        '''
        corr = future_df.apply(lambda x: Technical_Indicator_Tablib.pearson_corr(x, y), axis = 0)
        return corr

    @staticmethod
    def indicators_selected_by_Pearsonr(data, threshold = 0):
        '''

        :param data:the futures close price dataframe
        :param threshold: if abs(corr)>threshold,then keep,if the threshold is 0,then keep all the indicators
        :return: a dataframe with the indicators of all futures with price's corr>threshold

        '''
        indicators = Technical_Indicator_Tablib.get_all_indicators(data)
        # names = data.index.levels[0].to_list()
        corr_all = indicators.groupby('code').apply(
            lambda x: Technical_Indicator_Tablib.get_corr(x, data.loc[x.index, 'close']))

        corr_satisfy = corr_all[abs(corr_all) >= threshold]

        select = (corr_satisfy.apply(lambda x: x.dropna().index.to_list(), axis = 1)).to_frame()
        select.columns = ['indicators']

        return select

    @staticmethod
    def get_standardized_indicators_by_zscore(indicators):
        indicators_s = indicators.groupby('code').apply(
            lambda x: Technical_Indicator_Tablib.standardized_indicators_zscores(x))

        indicators_s.dropna(inplace = True)

        # drop the first 250 days data
        n = 250
        indicators_s = indicators_s.groupby('code').apply(lambda x: x.iloc[n:, :]).droplevel(level = 0)

        return indicators_s

    @staticmethod
    def get_normalization_indicators(indicators, dmax = 1, dmin = 0):
        indicators_n = indicators.groupby('code').apply(
            lambda x: (x - x.expanding().min()) * (dmax - dmin) / (x.expanding().max() - x.expanding().min()) + dmin)

        indicators_n.dropna(inplace = True)

        # drop the first 250 days data
        n = 250
        indicators_n = indicators_n.groupby('code').apply(lambda x: x.iloc[n:, :]).droplevel(level = 0)

        return indicators_n

