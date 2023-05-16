import efinance as ef
import re
import numpy as np
import pandas as pd

# use efinance to get data

info = ef.futures.get_futures_base_info()

# get main contract name and code

class Get_data:
    def __init__(self, begin_date = '19000101', end_date = '20500101'):
        '''

        :param begin_date: str, the starting date of the data. eg:'19000101' represents 1900-01-01
        :param end_date: str, the ending date of the data
        '''

        self.begin_date = begin_date
        self.end_date = end_date

    def get_all_main_future_data(self):
        pattern1 = re.compile(r'.*主力')
        pattern2 = re.compile(r'.*次主力')
        main_contract = info['期货名称'].apply(
            lambda x: re.findall(pattern1, x)[0] if re.findall(pattern1, x) and not re.findall(pattern2,
                                                                                               x) else np.nan).dropna()
        main_contract_info = info[info['期货名称'].isin(main_contract.values)]
        main_contract_info.to_csv("./main_contract_info.csv", encoding = 'gbk')

        # get all the data of the main contracts
        quote_ids = main_contract_info["行情ID"].to_list()
        futures_dict = ef.futures.get_quote_history(quote_ids, beg = self.begin_date, end = self.end_date)
        df = futures_dict[quote_ids[0]]
        for i in range(1, len(quote_ids)):
            df1 = futures_dict[quote_ids[i]]
            df = pd.concat([df, df1])
        df.drop(labels = ['换手率', '涨跌额'], inplace = True, axis = 1)
        df.columns = ["chi_name", 'code', 'date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude',
                      'change_percentage']
        df = df.reset_index(drop = True)
        return df


