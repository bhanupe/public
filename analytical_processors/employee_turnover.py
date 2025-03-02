import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from visualization.visualize import heat_map, hist_plot, bar_plot, count_plot
from wrangling.insights import explain, group_by_features
from sklearn.model_selection import train_test_split as split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score

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
        data3.drop(columns=['left', 'salary_encoded', 'department_encoded'], inplace=True)
        data3.to_csv('out.csv', index=False)
        print(data3.head(3))
        target_cols = data3.columns[-2:]
        print(target_cols)
        predictors = data3.drop(columns=target_cols)  # X
        target = data3[target_cols[1]]  # y
        print(predictors.shape)
        print(target.shape)
        # print(target)

        # X_train, X_test, y_train, y_test = split(predictors, target, test_size=0.30, random_state=12)
        # print(X_test.head(3))
        # print(y_test.head(3))
        # logreg = LogisticRegression(max_iter=5000)
        # logreg.fit(X_train, y_train)
        # pred_prob = logreg.predict_proba(X_test)
        # pred_class = logreg.predict(X_test)
        # for i in range(0, 3, 1):
        #     print(pred_prob[i], pred_class[i])

        ######### Splitting Data ##########
        X = data.drop(columns='average_montly_hours')
        y = data.average_montly_hours  # always a series
        print(f"x= ",X.shape)
        print(f"y= ",y.shape)

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

        ### Evaluate the model:
        r2Score = r2_score(y_test,pred_test_mlr)
        print(f"r2Score = ", r2Score)

        ### Plot the predictions
        plt.figure(figsize=(20,15))
        plt.scatter(y_test,pred_test_mlr)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('ACTUAL vs. Predicted')
        plt.show()

        ### print predicted values
        pred_df = pd.DataFrame({'Actual value' : y_test, 'Predicted value' : pred_test_mlr, 'Difference' : y_test - pred_test_mlr})
        print(f"Predicted values Data frame:")
        print(pred_df[0:20])




    except Exception as e:
        print(f"❌ Fatal error: {e}")
