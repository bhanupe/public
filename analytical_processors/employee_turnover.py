import copy
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from visualization.visualize import heat_map, hist_plot, count_plot
from wrangling.insights import explain

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

        explain(data)
        encoded_data = pd.get_dummies(data, dtype=int)
        print(encoded_data.head(3))
        encoded_data.to_csv('encoded_data.csv', index=False)
        target_cols = encoded_data['left']
        print(target_cols)

        ########### Correlation Matrix ############
        ## 2.1 Draw a heatmap of the correlation matrix between all numerical features or columns in the data.
        corr_mat = encoded_data.corr()
        mask = np.ones_like(corr_mat)
        mask[np.tril_indices_from(mask)] = 0

        # in the correlation matrix, value
        # The correlation values range from -1 to 1:
        # 1 means perfect positive correlation,
        # -1 means perfect negative correlation,
        # 0 means no correlation.
        # Based on a predefined threshold (for example, 0.9), you can identify pairs of columns that are highly correlated and drop them.
        print(f'Describe Correlation : \n{corr_mat}')
        heat_map(corr_mat)

        # Reasons behind employees leaving are influenced by
        # satisfaction_level -> salary -> work_accident -> department

        ########## Distribution plot ##############
        # Outcome of the Distributions:
        # Employee Satisfaction (satisfaction_level)
        # The distribution is bimodal, with peaks at low (around 0.1) and high (around 0.7-0.8) satisfaction levels.
        # This suggests that employees tend to be either very satisfied or very dissatisfied.
        hist_plot(encoded_data['satisfaction_level'])

        # Employee Evaluation (last_evaluation)
        # The distribution shows peaks around 0.5 and 0.85-1.0.
        # This suggests two groups: moderate performers and highly evaluated employees.
        hist_plot(encoded_data['last_evaluation'])

        # Employee Average Monthly Hours (average_monthly_hours)
        # The distribution is right-skewed, with peaks around 150-200 and 250+ hours.
        # This suggests two work-hour groups: regular workers and overworked employees.
        hist_plot(encoded_data['average_montly_hours'])
        ########## Distribution plot ##############

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
        # plt.show()

        ########## 2.3 Bar plot ##############
        # 2.3. Draw the bar plot of the employee project count of both employees
        # who left and stayed in the organization (use column number_project
        # and hue column left), and give your inferences from the plot.
        ########## Inference from countPlot is Employees with low number of projects which is 2 and Employees with high number of projects(7) left the company #################
        count_plot(encoded_data, 'number_project', 'left', 'Employee Project Count by Left/Stayed Status',
                   'Number of Projects',
                   'Count of Employees', "count")
        count_plot(encoded_data, 'number_project', 'left', 'Employee Project Percent by Left/Stayed Status',
                   'Number of Projects',
                   'Percentage of Employees', 'percent')
        count_plot(encoded_data, 'number_project', 'left', 'Employee Project Proportion by Left/Stayed Status',
                   'Number of Projects',
                   'Proportion of Employees', 'proportion')

        ## K-Means Clustering Steps
        ## 3.1 Choose columns satisfaction_level, last_evaluation, and left.
        test_data = encoded_data[encoded_data['left'] == 1][['satisfaction_level', 'last_evaluation']]
        test_data.head(2)

        ## Standardization using StandardScaler before K-Means Clustering
        sc = StandardScaler()
        scaled = sc.fit_transform(test_data)

        ## 3.2 Do K-means clustering of employees who left the company into 3 clusters?
        kmm = KMeans(n_clusters=3, random_state=12)
        kmm.fit(scaled)
        cluster_labels = kmm.predict(scaled)

        ## K-Means Cluster profiling
        cluster_data = copy.deepcopy(test_data)
        cluster_data['clus_label'] = cluster_labels
        print('----')
        print(cluster_data.head(3))

        ## Visualize Scatter Plot after K-Means clustering.
        f = plt.subplots(1, 1, figsize=(15, 6))
        sb.scatterplot(x=cluster_data.satisfaction_level, y=cluster_data.last_evaluation, hue=cluster_data.clus_label,
                       palette='rainbow_r')
        plt.xlabel('Satisfaction Level')
        plt.ylabel('Last Evaluation')
        plt.title('K-Means Clustering of Employees Who Left')
        plt.show()

        ## 3.3 Inference from Clusters
        ###### Cluster 0 (Low Satisfaction, High Evaluation) employees are at the highest risk—even if they perform well, they leave due to dissatisfaction.
        ###### Cluster 1 (High Satisfaction & Evaluation) might be leaving due to external offers or burnout.
        ###### Cluster 2 (Moderate Satisfaction & Evaluation) might indicate employees who were not particularly engaged.

        # 4.1 - Pre-process the data by converting categorical columns to numerical columns - Done Line #- 39
        # 4.2 -n Do the stratified split of the dataset to train and test in the ratio 80:20 with random_state=123

        X = encoded_data.drop(columns=['left'])  # Features
        y = encoded_data['left']  # Target variable

        # Step 3: Stratified Train-Test Split (80:20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)

        # Visualization before applying SMOTE
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sb.countplot(x=y_train, ax=axes[0])
        axes[0].set_title("Before SMOTE")
        axes[0].set_xlabel("Left (0 = Stayed, 1 = Left)")

        # Step 4: Handle Class Imbalance Using SMOTE
        smote = SMOTE(sampling_strategy='auto', random_state=123)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        sb.countplot(x=y_train_resampled, ax=axes[1])
        axes[1].set_title("After SMOTE")
        axes[1].set_xlabel("Left (0 = Stayed, 1 = Left)")

        plt.show()

        # Step 5: Verify Class Distribution Before & After
        print("Original Training Set Class Distribution:", dict(pd.Series(y_train).value_counts()))
        print("After SMOTE Class Distribution:", dict(pd.Series(y_train_resampled).value_counts()))

        print("X_train_resampled", X_train_resampled)
        print("y_train_resampled", y_train_resampled)

        print(y_train_resampled is None)  # Should be False
        print(type(y_train_resampled))  # Should be DataFrame or array

        # Step 5.1: Train a logistic regression model, apply a 5-fold CV, and plot the classification report.

        # Initialize the model
        logreg = LogisticRegression(max_iter=5000)

        # Create stratified k-fold cross-validator
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Perform cross-validated predictions
        y_pred_logreg = cross_val_predict(logreg, X_train_resampled, y_train_resampled, cv=skf)

        # Print classification report
        # 1. Precision -
        # Class 0 (Stayed): 81.6% of the employees predicted as “stayed” actually stayed.
        # Class 1 (Left): 80.2% of the employees predicted as “left” actually left.
        # Interpretation: model does a good job avoiding false positives in both classes.

        # 2. Recall - higher recall means we have recorded most of the people who are going to leave (if recall value is high then model is predicting correct)
        # Class 0: 79.8% of all actual “stayed” employees were correctly predicted.
        # Class 1: 82.1% of all actual “left” employees were correctly predicted.
        # Interpretation: model does slightly better at identifying who left, which is often more important in turnover prediction.

        # 3. F1-Score
        # Combines precision and recall into a balanced score.
        # Both classes have F1 around 0.81, meaning your model is performing fairly evenly and well across both classes.

        print("Classification Report for Logistic Regression (5-Fold CV):")
        print(classification_report(y_train_resampled, y_pred_logreg, digits=3))

        # 5.2 Train a Random Forest Classifier model, apply the 5-fold CV, and plot
        # the classification report.

        # Step 1: Create the model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Step 3: Generate predictions using 5-fold CV
        y_pred_rf = cross_val_predict(rf_model, X_train_resampled, y_train_resampled, cv=skf)

        # Step 4: Print classification report
        print("Classification Report for Random Forest (5-Fold CV):")
        print(classification_report(y_train_resampled, y_pred_rf, digits=3))

        # 5.3 Train a Gradient Boosting Classifier model, apply the 5-fold CV, and
        # plot the classification report.
        # Step 1: Create the model
        gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

        # Step 3: Perform cross-validated predictions
        y_pred_gb = cross_val_predict(gb_model, X_train_resampled, y_train_resampled, cv=skf)

        # Step 4: Classification report
        print("Classification Report for Gradient Boosting (5-Fold CV):")
        print(classification_report(y_train_resampled, y_pred_gb, digits=3))

        logreg.fit(X_train_resampled, y_train_resampled)
        rf_model.fit(X_train_resampled, y_train_resampled)
        gb_model.fit(X_train_resampled, y_train_resampled)

        # Find the ROC/AUC for each model and plot the ROC curve.
        log_reg_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
        rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
        gb_auc = roc_auc_score(y_test, gb_model.predict_proba(X_test)[:, 1])

        # Plot ROC curves
        log_fpr, log_tpr, _ = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
        rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
        gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_model.predict_proba(X_test)[:, 1])

        plt.figure(figsize=(8, 6))
        plt.plot(log_fpr, log_tpr, label=f'Logistic Regression (AUC = {log_reg_auc:.5f})')
        plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.5f})')
        plt.plot(gb_fpr, gb_tpr, label=f'Gradient Boosting (AUC = {gb_auc:.5f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison With X_train_resampled and y_train_resampled')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        log_reg = LogisticRegression(max_iter=5000)
        rfmodel = RandomForestClassifier(n_estimators=100, random_state=42)
        gbmodel = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

        log_reg.fit(X_train, y_train)
        rfmodel.fit(X_train, y_train)
        gbmodel.fit(X_train, y_train)

        # Find the ROC/AUC for each model and plot the ROC curve.
        logreg_auc = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
        rfauc = roc_auc_score(y_test, rfmodel.predict_proba(X_test)[:, 1])
        gbauc = roc_auc_score(y_test, gbmodel.predict_proba(X_test)[:, 1])

        # Plot ROC curves
        logfpr, logtpr, _ = roc_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
        rffpr, rftpr, _ = roc_curve(y_test, rfmodel.predict_proba(X_test)[:, 1])
        gbfpr, gbtpr, _ = roc_curve(y_test, gbmodel.predict_proba(X_test)[:, 1])

        plt.figure(figsize=(8, 6))
        plt.plot(logfpr, logtpr, label=f'Logistic Regression (AUC = {logreg_auc:.5f})')
        plt.plot(rffpr, rftpr, label=f'Random Forest (AUC = {rfauc:.5f})')
        plt.plot(gbfpr, gbtpr, label=f'Gradient Boosting (AUC = {gbauc:.5f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison With X_train and Y_tran')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Find the confusion matrix for each of the models.
        # Make predictions using each model
        logreg_preds = logreg.predict(X_test)
        rf_preds = rf_model.predict(X_test)
        gb_preds = gb_model.predict(X_test)

        # Logistic Regression
        print(":small_blue_diamond: Logistic Regression")
        print("Confusion Matrix:\n", confusion_matrix(y_test, logreg_preds))
        print("Classification Report:\n", classification_report(y_test, logreg_preds))

        # Random Forest
        print("\n:small_blue_diamond: Random Forest")
        print("Confusion Matrix:\n", confusion_matrix(y_test, rf_preds))
        print("Classification Report:\n", classification_report(y_test, rf_preds))

        # Gradient Boosting
        print("\n:small_blue_diamond: Gradient Boosting")
        print("Confusion Matrix:\n", confusion_matrix(y_test, gb_preds))
        print("Classification Report:\n", classification_report(y_test, gb_preds))

        # Explain which metric needs to be used from the confusion matrix:
        # Recall or Precision?
        # Recall - we have to use because with good performance and Leaving,then HR might be panic if recall is high)

        # 7.1 Using the best model, predict the probability of employee turnover
        # in the test data.

        # Train the best model (Random Forest)
        # rfmodel.fit(X_train, y_train)

        # Predict probabilities for class 1 (leaving)
        rf_probabilities = rfmodel.predict_proba(X_test)[:, 1]

        # Create a DataFrame to view predictions
        rf_prob_df = pd.DataFrame({
            'Employee_satisfaction_level': X_test["satisfaction_level"],
            'Employee_last_eval_level': X_test["last_evaluation"],
            'Employee_project': X_test["number_project"],
            'Employee_monthly_hrs': X_test["average_montly_hours"],
            'Employee_time_spend': X_test["time_spend_company"],
            'Employee_work_accident': X_test["Work_accident"],
            'Employee_promotion_last5': X_test["promotion_last_5years"],
            'Employee_dept': X_test["department_IT"],
            'Employee_rd': X_test["department_RandD"],
            'Employee_accounting': X_test["department_accounting"],
            'Employee_hr': X_test["department_hr"],
            'Employee_management': X_test["department_management"],
            'Employee_marketing': X_test["department_marketing"],
            'Employee_prod': X_test["department_product_mng"],
            'Employee_sales': X_test["department_sales"],
            'Employee_support': X_test["department_support"],
            'Employee_technical': X_test["department_technical"],
            'Employee_high_salary': X_test["salary_high"],
            'Employee_low_salary': X_test["salary_low"],
            'Employee_medium_salary': X_test["salary_medium"],
            'Actual': y_test.values,
            'Predicted_Probability': rf_probabilities
        })
        # Display the first few rows
        print(rf_prob_df.head(10))


        # Define a function to assign risk zone based on probability
        def assign_risk_zone(prob):
            if prob < 0.20:
                return 'Safe Zone (Green)'
            elif 0.20 < prob <= 0.60:
                return 'Low-Risk Zone (Yellow)'
            elif 0.60 < prob <= 0.90:
                return 'Medium-Risk Zone (Orange)'
            else:
                return 'High-Risk Zone (Red)'


        # Apply the risk zone categorization
        rf_prob_df['Risk_Zone'] = rf_prob_df['Predicted_Probability'].apply(assign_risk_zone)

        # Count number of employees in each risk zone
        risk_zone_counts = rf_prob_df['Risk_Zone'].value_counts().reset_index()
        risk_zone_counts.columns = ['Risk Zone', 'Employee Count']
        print(risk_zone_counts)

        # rf_prob_df.to_csv("employee_zone.csv",index=False)

        ## Clustering for employee data for the predicted probability who left and stayed

        ## K-Means Clustering Steps
        ## 3.1 Choose columns satisfaction_level, last_evaluation, and left.
        print(rf_prob_df.columns)
        outcome_data = rf_prob_df[['Employee_satisfaction_level', 'Employee_last_eval_level']]
        outcome_data.head(2)

        ## Standardization using StandardScaler before K-Means Clustering
        sc = StandardScaler()
        scaled_outcome = sc.fit_transform(outcome_data)

        ## 3.2 Do K-means clustering of employees who left the company into 3 clusters?
        kmm = KMeans(n_clusters=3, random_state=12)
        kmm.fit(scaled_outcome)
        cluster_labels = kmm.predict(scaled_outcome)

        ## K-Means Cluster profiling
        cluster_data = copy.deepcopy(outcome_data)
        cluster_data['clus_label'] = cluster_labels
        print('----')
        print(cluster_data.head(3))

        ## Visualize Scatter Plot after K-Means clustering.
        f1 = plt.subplots(1, 1, figsize=(15, 6))
        sb.scatterplot(x=cluster_data.Employee_satisfaction_level, y=cluster_data.Employee_last_eval_level,
                       hue=cluster_data.clus_label,
                       palette='rainbow_r')
        plt.show()

        ## Clustering for employee data for the predicted probability who is staying currently

        ## K-Means Clustering Steps
        ## 3.1 Choose columns satisfaction_level, last_evaluation, and left.
        # print(rf_prob_df.columns)

        outcome_data_stayed = rf_prob_df[rf_prob_df['Actual'] == 0][
            ['Employee_satisfaction_level', 'Employee_last_eval_level']]
        outcome_data_stayed.head(2)

        ## Standardization using StandardScaler before K-Means Clustering
        sc = StandardScaler()
        scaled_outcome = sc.fit_transform(outcome_data_stayed)

        ## 3.2 Do K-means clustering of employees who left the company into 3 clusters?
        kmm = KMeans(n_clusters=3, random_state=12)
        kmm.fit(scaled_outcome)
        cluster_labels = kmm.predict(scaled_outcome)

        ## K-Means Cluster profiling
        cluster_data = copy.deepcopy(outcome_data_stayed)
        cluster_data['clus_label'] = cluster_labels
        print('----')
        print(cluster_data.head(3))

        ## Visualize Scatter Plot after K-Means clustering.
        f2 = plt.subplots(1, 1, figsize=(15, 6))
        sb.scatterplot(x=cluster_data.Employee_satisfaction_level, y=cluster_data.Employee_last_eval_level,
                       hue=cluster_data.clus_label,
                       palette='rainbow_r')
        plt.show()

        ## Clustering for employee data for the predicted probability who is staying currently

        ## K-Means Clustering Steps
        ## 3.1 Choose columns satisfaction_level, last_evaluation, and left.
        # print(rf_prob_df.columns)

        outcome_data_stayed = rf_prob_df[rf_prob_df['Actual'] == 0][
            ['Employee_satisfaction_level', 'Employee_last_eval_level']]
        outcome_data_stayed.head(2)

        ## Standardization using StandardScaler before K-Means Clustering
        sc = StandardScaler()
        scaled_outcome = sc.fit_transform(outcome_data_stayed)

        ## 3.2 Do K-means clustering of employees who left the company into 3 clusters?
        kmm = KMeans(n_clusters=3, random_state=12)
        kmm.fit(scaled_outcome)
        cluster_labels = kmm.predict(scaled_outcome)

        ## K-Means Cluster profiling
        cluster_data = copy.deepcopy(outcome_data_stayed)
        cluster_data['clus_label'] = cluster_labels
        print('----')
        print(cluster_data.head(3))

        ## Visualize Scatter Plot after K-Means clustering.
        f2 = plt.subplots(1, 1, figsize=(15, 6))
        sb.scatterplot(x=cluster_data.Employee_satisfaction_level, y=cluster_data.Employee_last_eval_level,
                       hue=cluster_data.clus_label,
                       palette='rainbow_r')
        plt.show()

        ## Clustering for employee data for the predicted probability who is staying currently and high risk zone

        ## K-Means Clustering Steps
        ## 3.1 Choose columns satisfaction_level, last_evaluation, and left.
        # print(rf_prob_df.columns)

        outcome_data_stayed_high = \
            rf_prob_df[rf_prob_df['Actual'] == 0][rf_prob_df['Risk_Zone'] == 'High-Risk Zone (Red)'][
                ['Employee_satisfaction_level', 'Employee_last_eval_level']]
        outcome_data_stayed_high.head(2)

        ## Standardization using StandardScaler before K-Means Clustering
        sc = StandardScaler()
        scaled_outcome = sc.fit_transform(outcome_data_stayed_high)

        ## 3.2 Do K-means clustering of employees who left the company into 3 clusters?
        kmm = KMeans(n_clusters=3, random_state=12)
        kmm.fit(scaled_outcome)
        cluster_labels = kmm.predict(scaled_outcome)

        ## K-Means Cluster profiling
        cluster_data = copy.deepcopy(outcome_data_stayed_high)
        cluster_data['clus_label'] = cluster_labels
        print('----')
        print(cluster_data.head(3))

        ## Visualize Scatter Plot after K-Means clustering.
        f2 = plt.subplots(1, 1, figsize=(15, 6))
        sb.scatterplot(x=cluster_data.Employee_satisfaction_level, y=cluster_data.Employee_last_eval_level,
                       hue=cluster_data.clus_label,
                       palette='rainbow_r')
        plt.show()



    except Exception as e:
        print(f"error={e}, traceback={traceback.format_exc()}, line_no={e.__traceback__.tb_lineno}")
