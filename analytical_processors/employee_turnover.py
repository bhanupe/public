import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from visualization.visualize import heat_map, hist_plot, bar_plot
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
        # The correlation values range from -1 to 1:
        # a value of 1 means perfect positive correlation,
        # -1 means perfect negative correlation,
        # and 0 means no correlation.
        # Based on a predefined threshold (for example, 0.9), you can identify pairs of columns that are highly correlated and drop them.
        print(f'Describe Correlation : \n{corr_mat}')
        heat_map(corr_mat)
        # Reasons behind employees leaving are influenced by
        # satisfaction_level -> salary -> work_accident -> department
        ########### Correlation Matrix ############

        ########## Distribution plot ##############
        #Outcome of the Distributions:
# Employee Satisfaction (satisfaction_level)
# The distribution is bimodal, with peaks at low (around 0.1) and high (around 0.7-0.8) satisfaction levels.
# This suggests that employees tend to be either very satisfied or very dissatisfied.

# Employee Evaluation (last_evaluation)
# The distribution shows peaks around 0.5 and 0.85-1.0.
# This suggests two groups: moderate performers and highly evaluated employees.

# Employee Average Monthly Hours (average_monthly_hours)
# The distribution is right-skewed, with peaks around 150-200 and 250+ hours.
# This suggests two work-hour groups: regular workers and overworked employees.

        hist_plot(data['satisfaction_level'])
        hist_plot(data['last_evaluation'])
        hist_plot(data['average_montly_hours'])

# Filter the data for employees who left
        df_left = data[data["left"] == 1]
        columns_to_plot = ["satisfaction_level", "last_evaluation", "average_montly_hours"]
        # Plot the distribution for employees who left
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(columns_to_plot, 1):
            plt.subplot(1, 3, i)
            sb.histplot(df_left[col], kde=True, bins=30, color="red")
            plt.title(f"Distribution of {col} (Employees who left)")
        plt.tight_layout()
        plt.show()

        ########## Bar plot ##############
        # bar_plot(data, 'number_project', 'left', 'analysis')


    except Exception as e:
        print(f"❌ Fatal error: {e}")
