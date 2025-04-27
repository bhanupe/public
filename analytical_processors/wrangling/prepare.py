import pandas as pd


def fill_null(data, column, value):
    return data.fillna({column: value}, inplace=True)


def convert_data_type(data, column, dest_type):
    if dest_type == 'datetime':
        data.Date = pd.to_datetime(column)
        return data


def encode(data, datatype):
    data_encoded = pd.get_dummies(data, dtype=datatype)
    data_encoded.to_csv('../data/data_encoded.csv', index=False)
    return data_encoded
