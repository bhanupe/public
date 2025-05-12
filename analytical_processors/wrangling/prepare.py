import pandas as pd
from scipy import stats
import numpy as np


def fill_null(data, column, value):
    return data.fillna({column: value}, inplace=True)


def convert_data_type(data, column, dest_type):
    if dest_type == 'datetime':
        data.Date = pd.to_datetime(column)
        return data


def encode(data, datatype, outputfile=None):
    data_encoded = pd.get_dummies(data, dtype=datatype)
    if outputfile:
        data_encoded.to_csv(f'{outputfile}', index=False)
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


def add_value_from_natural_log(data):
    # Example natural logarithm value
    ln_value = 10.9861
    # Calculate the original value using exp()
    original_value = np.exp(ln_value)
    print("Original value:", original_value)


def check_class_imbalance(df, target_column, imbalance_threshold=0.10):

    target_counts = df[target_column].value_counts()
    total = target_counts.sum()
    percentages = (target_counts / total) * 100

    print("\nClass Distribution:")
    for label, count in target_counts.items():
        print(f"Class {label}: {count} samples ({percentages[label]:.2f}%)")

    # Get majority and minority percentages
    majority_class_percentage = max(percentages)
    minority_class_percentage = min(percentages)

    # Check imbalance
    if abs(majority_class_percentage - minority_class_percentage) > (imbalance_threshold * 100):
        print("\n🚨 Data is IMBALANCED.")
    else:
        print("\n✅ Data is BALANCED.")