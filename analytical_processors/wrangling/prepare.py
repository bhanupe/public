import pandas as pd
from scipy import stats


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


def remove_duplicates(data):
    # Keep only unique rows based on all columns
    return data.drop_duplicates()


def add_zscores(data):
    # Calculate Z-scores for each numerical column
    # Add Z-score columns to the original DataFrame
    # Interpretation of Z-scores:
    # Z-score = 0: The data point is exactly equal to the mean.
    # Z-score > 0: The data point is above the mean.
    # Z-score < 0: The data point is below the mean.
    # Magnitude of Z-score: The magnitude of the Z-score indicates how far the data point is from the mean.
    # A larger magnitude indicates a greater distance.
    z_scores = data.select_dtypes(include=['number']).apply(lambda x: stats.zscore(x))
    data_with_zscores = data.join(z_scores.add_suffix('_zscore'))
    return z_scores, data_with_zscores
