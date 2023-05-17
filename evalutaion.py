import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np

# outcome = pd.read_csv("./prediction for am.rolling_pre.csv", index_col = [0, 1], parse_dates = [1])
image_path = "./Image/"


class Evaluation:
    @staticmethod
    def plot_nav(outcome: DataFrame):
        '''

        :param outcome: with index [symbol,date]
        :return:
        '''
        nav = outcome.apply(lambda x: (1 + x / 100).cumprod())
        nav.droplevel(0).plot()
        plt.legend()
        plt.title(nav.index[0][0])
        plt.savefig(image_path + nav.index[0][0]+" prediction nav.png")

    @staticmethod
    def get_square_error(outcome: DataFrame):
        '''
        get return prediction mean square error

        :param outcome: two column, one is prediction and one is true value
        :return:
        '''
        # because is measure in 100%
        err = np.sqrt((((outcome.iloc[:, 0] - outcome.iloc[:, 1]) / 100) ** 2).mean())
        error = pd.DataFrame(index = [outcome.index[0][0]], data = err, columns = ['mean square error'])
        return error
