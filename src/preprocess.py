import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import yaml


def preprocess(data_path):
    
    # read excel-file
    df = pd.read_excel(data_path)

    # change column names to better understand it
    df.columns = df.iloc[0]
    df = df[1:]

    # make all features int-types
    df = df[:].astype(np.int64)

    # How many time the overdue was
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df["num_late_months"] = (df[pay_cols] > 0).sum(axis=1)

    # BILL_AMT6 / LIMIT_BAL 
    # the closer to 1, the higher the risk of default
    df['debt_ratio'] = df['BILL_AMT6'] / df['LIMIT_BAL']

    # all PAY_AMT / all BILL_AMT
    # if payment_ratio = 1 - a client pays his bills fully
    # if payment_ratio > 1 - a client pays more than his debt
    # if payment_ratio < 1 - a client pays less than his debt
    bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    pay_amt_cols  = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
    df['payment_ratio'] = df[pay_amt_cols].sum(axis=1) / df[bill_cols].sum(axis=1)

    return df


if __name__ == '__main__':
    params = yaml.safe_load(open('params.yaml'))['preprocess']
    
    df = preprocess(params['input'])

    df.to_csv(params['output'])

    print(f'Processed dataset: {params['output']}')

    

