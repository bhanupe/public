import pandas as pd


def printline():
    print('-----------------------------------------------------------------------------------------------------------')


def explain(data):
    pd.set_option('display.max_columns', None)
    # Disable word wrapping when printing DataFrames
    pd.set_option('display.max_colwidth', None)
    # Adjust the display width to accommodate all columns
    pd.set_option('display.width', 500)  # You can adjust the value as needed
    printline()
    print(f'Shape : \n{data.shape}')
    printline()
    print(f'Row labels : \n{data.index}')
    printline()
    print(f'Column names : \n{data.columns}')
    printline()
    print(f'Data type : \n{data.dtypes}')
    printline()
    print(f'Data info : \n{data.info()}')
    printline()
    print(f'Describe data : \n{data.describe()}')
    printline()
    # this returns if there is at least one null value in any of the column data
    print(f'At least one null value: \n{data.isna().sum(axis=0)}')
    printline()
    missing_val = data.isna().sum(axis=0)
    print(f'Column which has null is: \n{missing_val[missing_val == 1]}')


def group_by_features(data):
    # %% d. Share your insights regarding the application of the GroupBy() function for
    # either data chunking or merging, and offer a recommendation based on
    # your analysis.
    # Grouping the data by 'State' and 'Time' 'Group' to analyze total sales and units sold
    data_grouped_group = data.groupby(['State', 'Time', 'Group']).agg({'Sales': 'sum', 'Unit': 'sum'}).reset_index()
    sorted_group = data_grouped_group.sort_values(by='Sales', ascending=False)
    print('Grouping the data by State and Time Group to analyze total sales and units sold')
    print(sorted_group)
    # Applying standard deviation to the 'Sales' column and creating a new column
    print('Applying standard deviation to the Sales column and creating a new column')
    data["Sales_Std_Dev"] = data.groupby(["State", "Time"])["Sales"].transform("std")
    print(data)
    print(data.select_dtypes(include='int').describe())
    print('Mode of Data')
    print(data.select_dtypes(include='int').mode())
    return data
