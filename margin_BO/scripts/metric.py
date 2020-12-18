"""
2020 Summer internship

implement metric for porfolio
"""
import pandas as pd
import numpy as np

class metric:

    @staticmethod
    def change_quantity(starting_portfolio: pd.DataFrame,
                        optimal_allocation: pd.DataFrame
                        ):
        
        diff = abs((starting_portfolio.to_numpy() - optimal_allocation.to_numpy())).sum(axis=1)/2
        
        
        return pd.DataFrame(diff, index=starting_portfolio.index, columns=["change quantity"])
    
    @staticmethod
    def risk_measurement(optimal_allocation: pd.DataFrame
                        ):
        two_year_risk = ["BTSU0", "DUU0"]
        five_year_risk = ["OEU0"]
        ten_year_risk = ["IKU0", "OATU0", "RXU0"]
        thirty_year_risk = ["UBU0"]

        two_year = []
        five_year = []
        ten_year = []
        thirty_year = []


        accounts = optimal_allocation.columns

        for index,row in optimal_allocation.iterrows():
            if index in two_year_risk:
                two_year.append(row.to_numpy()*2)
            elif index in five_year_risk:
                five_year.append(row.to_numpy()*5)
            elif index in ten_year_risk:
                ten_year.append(row.to_numpy()*10)
            elif index in thirty_year_risk:
                thirty_year.append(row.to_numpy()*30)
        index = ["2_year", "5_year", "10_year", "30_year"]
        
        two_year = sum(two_year).reshape(-1,1)
        five_year = sum(five_year).reshape(-1,1)
        ten_year = sum(ten_year).reshape(-1,1)
        thirty_year = sum(thirty_year).reshape(-1,1)

        all_risk = np.concatenate([two_year, five_year, ten_year, thirty_year], axis=1).T

        return pd.DataFrame(all_risk, columns=accounts, index=index)