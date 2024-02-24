# general imports

# conda packages
import os

import pandas as pd
import numpy as np

# local imports


def add_date_columns(df):
    """
    Add a yearmonth string and pd.DataTime columns to the dataframe.
    """
    df = df.copy()
    
    # create month number from 0 to 31
    yearmonths = list(np.sort(np.unique(df.yearmonth)))
    df['yearmonth_nr'] = df.yearmonth.map(lambda ym: yearmonths.index(ym))
    
    # create year and month columns
    year = df.yearmonth.astype(str).str.slice(0, 4)
    df['year'] = year.astype(int)
    month = df.yearmonth.astype(str).str.slice(4, 6)
    df['month'] = month.astype(int)
    
    # create string column of yearmonth
    df['yearmonth_str'] = year + "-" + month
    
    # creat pd.DataTime column for yearmonth
    df['yearmonth_dt'] = pd.to_datetime(df.yearmonth, format='%Y%m')
    return df


def train_val_test_split(df: pd.DataFrame, nr_train: int, nr_val: int, nr_test: int):
    """
    Split the dataset into trian/val/split by intervals of months.
    """
    if nr_train + nr_val + nr_test != np.max(df.yearmonth_nr) + 1:
        raise ValueError(
            f'Incorrect train/test/val split. Total months in split:' + \
            f'{nr_train + nr_val + nr_test} != {np.max(df.yearmonth_nr) + 1}'
        )
    split_1 = nr_train
    split_2 = nr_train + nr_val 
    split_3 = nr_train + nr_val + nr_test
    
    df_train = df[df.yearmonth_nr.isin(np.arange(0, split_1))]
    df_val = df[df.yearmonth_nr.isin(np.arange(split_1, split_2))]
    df_trainval = df[df.yearmonth_nr.isin(np.arange(0, split_2))]
    df_test = df[df.yearmonth_nr.isin(np.arange(split_2, split_3))]
    
    return df_train, df_val, df_trainval, df_test


def find_incomplete_clients(df_applications, df_customers):
    """
    Find those clients of which credit information is incomplete.
    These are deleted from the dataset.
    """
    clients = np.unique(df_applications.client_nr)
    nr_months = np.unique(df_applications.yearmonth).shape[0]

    month_counts_cust = df_customers.groupby('client_nr').yearmonth.count()
    month_counts_appl = df_applications.groupby('client_nr').yearmonth.count()

    missing_in_cust = month_counts_cust < nr_months
    missing_in_appl = month_counts_appl < nr_months

    incomplete_clients = missing_in_cust | missing_in_appl
    incomplete_clients = incomplete_clients[incomplete_clients].index.to_numpy()

    complete_clients = clients[~np.isin(clients, incomplete_clients)]

    return complete_clients, incomplete_clients


def get_data(nr_train, nr_val, nr_test):
    """
    Load mortgage data from csv's in a single DataFrame.
    """
    # Load dataframes
    df_customers = pd.read_csv(os.path.join('data', 'customers.csv'))
    df_applications = pd.read_csv(os.path.join('data', 'credit_applications.csv'))
    # df_applications.drop('Unnamed: 0', inplace=True)
    # df_customers.drop('Unnamed: 0', inplace=True)

    complete_clients, incomplete_clients = find_incomplete_clients(df_applications, df_customers)

    df = df_applications.merge(df_customers, on=['client_nr', 'yearmonth'], how='inner')
    df = df[df.client_nr.isin(complete_clients)]

    # cast to float for floating point predictions
    df.nr_credit_applications = df.nr_credit_applications.astype(float)
    df['has_credit_applications'] = (df.nr_credit_applications >= 1.0).astype(float)

    # Add further date columns
    df = add_date_columns(df)

    # Add categorical variables
    df['client_nr_cat'] = df.client_nr.apply(lambda s: 'client_' + str(s))
    df['month_cat'] = df.month.apply(lambda s: 'month_' + str(s))

    # make train/val/test splits
    df_train, df_val, df_tv, df_test = train_val_test_split(
        df, nr_train, nr_val, nr_test
    )

    regressors = [
        # 'total_nr_trx',
        'nr_debit_trx',
        'nr_credit_trx',
        'volume_debit_trx',
        'volume_credit_trx',
        'min_balance',
        'max_balance',
        # 'CRG',
    ]

    return (df_train, df_val, df_tv, df_test, df,
            complete_clients, incomplete_clients,
            regressors)
