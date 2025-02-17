import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from visualization.visualize import heat_map, hist_plot, bar_plot, count_plot
from wrangling.insights import explain, group_by_features
from sklearn.model_selection import train_test_split as split
from sklearn.linear_model import LinearRegression

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

        ########## 2.3 Bar plot ##############
        # 2.3. Draw the bar plot of the employee project count of both employees
        # who left and stayed in the organization (use column number_project
        # and hue column left), and give your inferences from the plot.
        ########## Inference from countPlot is Employees with low number of projects which is 2 and Employees with high number of projects(7) left the company #################
        count_plot(data, 'number_project', 'left', 'Employee Project Count by Left/Stayed Status', 'Number of Projects',
                   'Count of Employees', "count")
        count_plot(data,'number_project','left','Employee Project Percent by Left/Stayed Status','Number of Projects',
                   'Percentage of Employees','percent')
        count_plot(data, 'number_project', 'left', 'Employee Project Proportion by Left/Stayed Status', 'Number of Projects',
                   'Proportion of Employees', 'proportion')

        ######### Splitting Data ##########
        X = data.drop(columns=['average_montly_hours'],axis = 1).values # dataframe
        y = data['average_montly_hours'].values  # always a series
        print(f"x= ",X)
        print(f"y= ",y)

        X_train, X_test, y_train, y_test = split(X, y, test_size=0.2, random_state=12) ## split train and test data by 80 and 20 percent

        print(f"X_train no. of rows = ", X_train.shape[0])
        print(f"y_train no. of rows = ", y_train.size)

        print(f"X_test no. of rows  = ", X_test.shape[0])
        print(f"y_test no. of rows  = ", y_test.size)

        mlr_mod = LinearRegression()
        mlr_mod.fit(X_train, y_train) ## fit and train the model

        ####predict for first row
        data.head(2);
        y_pred = mlr_mod.predict([[0.38,0.53,2,3,0,1,0,1,1]])
        print(f"y_pred for first row = ", y_pred ,"but actual value of y= 157" )

        y_pred2 = mlr_mod.predict([[0.8,0.86,5,6,0,1,0,1,2]])
        print(f"y_pred for second row = ", y_pred2,"but actual value of y= 262")


        ###### predict for Test and Train
        pred_train_mlr = mlr_mod.predict(X_train)
        pred_test_mlr = mlr_mod.predict(X_test)

        print(f"pred_train_mlr = ", pred_train_mlr)
        print(f"pred_test_mlr = ",pred_test_mlr)

        print(f"pred_train_mlr :5 = ", pred_train_mlr[:5])
        print(f"pred_test_mlr :5 = ", pred_test_mlr[:5])

    except Exception as e:
        print(f"❌ Fatal error: {e}")
