import pandas as pd


def explain(data):
    print(f'Shape : \t{data.shape}')
    print(f'Row labels : \t{data.index}')
    print(f'Column names : \n{data.columns}')
    print(f'Data type : \n{data.dtypes}')
    pd.set_option('display.max_columns',None)
    print(f'Data info : \n{data.info()}')
    print(f'Describe data : \n{data.describe()}')
    # this returns if there is at least one null value in any of the column data
    print(data.isna().sum(axis=0))
    missing_val = data.isna().sum(axis=0)
    print('column which has null is')
    print(missing_val[missing_val == 1])


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
