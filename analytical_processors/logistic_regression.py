import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from visualization.visualize import heat_map, hist_plot, bar_plot, count_plot
from wrangling.insights import explain, group_by_features
from sklearn.model_selection import train_test_split as split
from sklearn.linear_model import LinearRegression, LogisticRegression

# Main Execution
if __name__ == "__main__":
    try:
        file_path = '../data/HR_comma_sep.csv'
        data = pd.read_csv(file_path)
        ################### EDA ###################
        ########### Data Quality Checks ###########
        # Perform data quality checks by checking for missing values, if any.
        explain(data)
        # One of the column was with wrong name 'sales'; renamed in the dataframe to 'department'
        data.rename(columns={'sales': 'department'}, inplace=True)
        explain(data)
        print(data.head(3))
        ########### Data Quality Checks ###########

        ########### Correlation Matrix ############
        # Draw a heatmap of the correlation matrix between all numerical features or columns in the data.
        # Numerical mapping for salary levels
        salary_mapping = {
            "low": 1,
            "medium": 2,
            "high": 3
        }

        # Numerical mapping for departments
        department_mapping = {
            "sales": 1,
            "accounting": 2,
            "hr": 3,
            "technical": 4,
            "support": 5,
            "management": 6,
            "IT": 7,
            "product_mng": 8,
            "marketing": 9,
            "RandD": 10
        }

        # Apply salary mapping
        data['salary_encoded'] = data["salary"].map(salary_mapping)
        # Apply department mapping
        data['department_encoded'] = data["department"].map(department_mapping)
        explain(data)
        data2 = pd.get_dummies(data, dtype=int)
        data2['left_str'] = 'no'
        data2.loc[data2['left'] == 1, 'left_str'] = 'yes'
        data3 = pd.get_dummies(data2, dtype=int)
        data3.drop(columns=['left','salary_encoded','department_encoded'], inplace=True)
        data3.to_csv('out.csv', index=False)
        print(data3.head(3))
        target_cols = data3.columns[-2:]
        print(target_cols)
        predictors = data3.drop(columns=target_cols) # X
        target = data3[target_cols[1]] # y
        print(predictors.shape)
        print(target.shape)
        # print(target)

        X_train, X_test, y_train, y_test = split(predictors, target, test_size=0.30, random_state=12)
        print(X_test.head(3))
        print(y_test.head(3))
        logreg = LogisticRegression(max_iter=5000)
        logreg.fit(X_train, y_train)
        pred_prob = logreg.predict_proba(X_test)
        pred_class = logreg.predict(X_test)
        for i in range(0,3,1):
            print(pred_prob[i], pred_class[i])
    except Exception as e:
        print(f"❌ Fatal error: {e}")