import pandas as pd
import numpy as np

from visualization.visualize import heat_map
from wrangling.insights import explain, group_by_features

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

        # Remove all object type data values
        data.drop(columns={'department', 'salary'}, inplace=True)
        explain(data)

        corr_mat = data.corr()

        mask = np.ones_like(corr_mat)
        mask[np.tril_indices_from(mask)] = 0

        # in the correlation matrix, value
        # 0 -> No correlation
        # +ve -> The variables are proportional to each other
        # -ve -> The variables are inversely proportional to each other
        print(f'Describe Correlation : \n{corr_mat}')
        heat_map(corr_mat)
        # Reasons behind employees leaving are influenced by
        # satisfaction_level -> salary -> work_accident -> department
        ########### Correlation Matrix ############


    except Exception as e:
        print(f"❌ Fatal error: {e}")
