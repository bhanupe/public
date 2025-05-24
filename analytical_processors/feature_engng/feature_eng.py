import numpy as np


def drop_highly_correlated_features(df, threshold=0.85):
    # 1. Compute correlation matrix
    corr_matrix = df.corr().abs()

    # 2. Create an upper triangle matrix of correlations
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # 3. Find columns with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"Columns to drop (correlation > {threshold}): {to_drop}")

    # 4. Drop those columns from DataFrame
    df_reduced = df.drop(columns=to_drop)

    return df_reduced
