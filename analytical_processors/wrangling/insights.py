import pandas as pd


def printline():
    print('-----------------------------------------------------------------------------------------------------------')


def explain(data):
    # Exploratory Data Analysis (EDA) is like the detective work you do with your data before building a model — you explore, summarize, and understand it. 🔍
    # Here are the main activities involved in EDA:
    #
    # 1. Understanding the Data Structure
    #   - Check the shape (rows × columns), data types (numeric, categorical, datetime, etc.).
    #   - Look at sample records (e.g., using `.head()` in pandas).
    #
    # 2. Checking for Missing Values
    #   - Identify where data is missing.
    #   - Decide whether to fill them, drop them, or leave them.
    #
    # 3. Summary Statistics
    #   - Mean, median, standard deviation, min, max, quantiles.
    #   - Helps you understand central tendency and spread.
    #
    # 4. Univariate Analysis (single variable)
    #   - Histograms, box plots, bar plots.
    #   - Understand distribution, skewness, outliers.
    #
    # 5. Bivariate/Multivariate Analysis (two or more variables)
    #   - Scatter plots, heatmaps (correlation matrices), pair plots.
    #   - Study relationships between features.
    #
    # 6. Data Cleaning
    #   - Correct inconsistencies (e.g., wrong data types, typos).
    #   - Remove duplicates.
    #
    # 7. Outlier Detection
    #   - Find and handle extreme values (with boxplots, z-scores, etc.).
    #
    # 8. Feature Engineering (early stage)
    #   - Create new features or modify existing ones based on early insights.
    #
    # 9. Visualization
    #   - Use graphs and charts to "see" patterns and problems (matplotlib, seaborn, etc.).
    #   - Visual tools make hidden trends pop out.
    #
    # 10. Correlation Analysis
    #    - Identify how strongly variables are related (important for model building later).
    #

    pd.set_option('display.max_columns', None)
    # Disable word wrapping when printing DataFrames
    pd.set_option('display.max_colwidth', None)
    # Adjust the display width to accommodate all columns
    pd.set_option('display.width', 500)  # You can adjust the value as needed

    printline()
    print(f'Shape : \n{data.shape}')

    printline()
    print(f'Data type : \n{data.dtypes}')

    printline()
    print(f'Row labels : \n{data.index}')

    printline()
    print(f'Column names : \n{data.columns}')

    printline()
    print(f'Data info : \n{data.info()}')

    printline()
    print(f'Describe data : \n{data.describe()}')

    printline()
    # Check missing values
    print(f'At least one null value method null : \n{data.isnull().sum()}')

    printline()
    # this returns if there is at least one null value in any of the column data
    print(f'At least one null value method na : \n{data.isna().sum(axis=0)}')

    printline()
    missing_val = data.isna().sum(axis=0)
    print(f'Column which has null is: \n{missing_val[missing_val == 1]}')

    printline()
    # Find duplicates based on all columns
    duplicates = data.duplicated()
    # Display the boolean Series
    print(f'Duplicate data: \n{duplicates}')

    printline()
    # Display duplicate rows
    duplicate_rows = data[duplicates]
    print(f'Duplicate rows: \n{duplicate_rows}')

    printline()
    print(data.head(5))


def group_by_features(data, by, sort_column, aggregation_settings=None):
    if aggregation_settings:
        # %% d. Share your insights regarding the application of the GroupBy() function for
        # either data chunking or merging, and offer a recommendation based on
        # your analysis.
        # Grouping the data by 'State' and 'Time' 'Group' to analyze total sales and units sold
        data_grouped_group = data.groupby(by).agg(aggregation_settings).reset_index()
        sorted_group = data_grouped_group.sort_values(by=sort_column, ascending=False)
        print('Grouping the data by State and Time Group to analyze total sales and units sold')
        print(sorted_group)
        return sorted_group
    else:
        # Applying standard deviation to the 'Sales' column and creating a new column
        print('Applying standard deviation to the Sales column and creating a new column')
        data["Sales_Std_Dev"] = data.groupby(by)[sort_column].transform("std")
        print(data)
        print(data.select_dtypes(include='int').describe())
        print('Mode of Data')
        print(data.select_dtypes(include='int').mode())
        return data
