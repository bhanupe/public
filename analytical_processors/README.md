## Modules
### requirements.txt
```
pandas==2.2.3
seaborn==0.13.2
scikit-learn==1.6.1
```

#### Environment preparation
please install the requirements 
```
python -m pip install --upgrade pip
python install -r requirements.txt
```

### Sales Assignment - main.py
```
import pandas as pd

from analytical_processors.analysis.data_analysis import sales, sales_normalization, sales_by_time
from analytical_processors.visualization.visualize import bar_plot, line_plot
from analytical_processors.wrangling.insights import explain, group_by_features
from analytical_processors.wrangling.normalization import min_max_normalization
from analytical_processors.wrangling.prepare import fill_null, convert_data_type

# Main Execution
if __name__ == "__main__":
    try:
        file_path = '../data/AusApparalSales4thQrt2020.csv'
        data = pd.read_csv(file_path)
        explain(data)
        # fill_null(data,'Time','Morning')
        # fill_null(data, 'Sales', 20000)
        explain(data)
        data = convert_data_type(data,data.Date,'datetime')
        print('After converting datatime datatype')
        print(data.info())
        min_max_normalization(data)
        group_by_features(data)
        grouped_data_by_state = sales(data,'State')
        print( f"Maximum Sales in State={grouped_data_by_state.idxmax()}")
        print(f"Minimum Sales in State={grouped_data_by_state.idxmin()}")
        grouped_data_by_group = sales(data, 'Group')
        print(f"Maximum Sales by Group={grouped_data_by_group.idxmax()}")
        print(f"Minimum Sales by Group={grouped_data_by_group.idxmin()}")

        grouped_data_by_state_normalization = sales_normalization(data, 'State')
        print(f"Maximum Sales using normalization in State={grouped_data_by_state_normalization.idxmax()}")
        print(f"Minimum Sales using normalization in State={grouped_data_by_state_normalization.idxmin()}")
        grouped_data_by_group_normalization = sales_normalization(data, 'Group')
        print(f"Maximum Sales using normalization by Group={grouped_data_by_group_normalization.idxmax()}")
        print(f"Minimum Sales using normalization by Group={grouped_data_by_group_normalization.idxmin()}")

        monthly_report = sales_by_time(data, 'month')
        print(f"Monthly Sales={data.groupby(monthly_report)['Sales'].sum()}")
        weekly_report = sales_by_time(data, 'day_of_week')
        print(f"Weekly Sales={data.groupby(weekly_report)['Sales'].sum()}")
        quarterly_report = sales_by_time(data, 'quarter')
        print(f"Quarterly Sales={data.groupby(quarterly_report)['Sales'].sum()}")

        monthly_report_sales_normalization = sales_by_time(data, 'month')
        print(f"Monthly sales_normalization={data.groupby(monthly_report)['sales_normalization'].sum()}")
        weekly_report_sales_normalization = sales_by_time(data, 'day_of_week')
        print(f"Weekly sales_normalization={data.groupby(weekly_report)['sales_normalization'].sum()}")
        quarterly_report_sales_normalization = sales_by_time(data, 'quarter')
        print(f"Quarterly sales_normalization={data.groupby(quarterly_report)['sales_normalization'].sum()}")

        bar_plot(data, 'State', 'Sales', 'Group')
        bar_plot(data, 'Group', 'Sales', 'State')
        sorted_data = data.sort_values(['Time'], ascending=False)
        line_plot(sorted_data.groupby('Time')['Sales'].sum().reset_index())
    except Exception as e:
        print(f"❌ Fatal error: {e}")
```

### Wrangling
#### wrangling/insights.py
```
import pandas as pd

def explain(data):
    print(f'Shape : \t{data.shape}')
    print(f'Row Labels : \t{data.index}')
    print(f'Column Names : \n{data.columns}')
    print(f'DataType : \n{data.dtypes}')
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
    print(data_grouped_group)
    sorted_group = data_grouped_group.sort_values(by='Sales', ascending=False)
    print('Grouping the data by State and Time Group to analyze total sales and units sold')
    print(sorted_group)
    # Applying standard deviation to the 'Sales' column and creating a new column
    print('Applying standard deviation to the Sales column and creating a new column')
    data["Sales_Std_Dev"] = data.groupby(["State", "Time"])["Sales"].transform("std")
    print(data)
    print(data.select_dtypes(include = 'int' ).describe())
    print('Mode of Data')
    print(data.select_dtypes(include = 'int').mode())
    return data
```
#### wrangling/prepare.py
```
import pandas as pd
def fill_null(data,column,value):
    return  data.fillna({column: value}, inplace = True)
def convert_data_type(data,column,dest_type):
    if dest_type == 'datetime':
        data.Date = pd.to_datetime(column)
        return data
```
#### wrangling/normalization.py
```
from sklearn.preprocessing import MinMaxScaler

# TBD C. Choose a suitable data wrangling technique—either data standardization
# or normalization. Execute the preferred normalization method and
# present the resulting data. (Normalization is the preferred approach for this
# problem.)  Normalization (Min-Max Scaling)	When data needs to be in a fixed range (0 to 1)	Xnormalized=Xmax−XminX−Xmin
def min_max_normalization(data):
    scaler = MinMaxScaler()
    # Apply normalization to 'Sales' column
    data['sales_normalization'] = scaler.fit_transform(data[['Sales']])
    print(data[['Unit','Sales','sales_normalization']].agg(['min','max']))
    return data
```

### Analysis
#### analysis/data_analysis.py
```
# Identify the group with the highest sales and the group with the lowest
# sales based on the data provided.

def sales(data,column):
    grouped_data = data.groupby(column)
    return grouped_data['Sales'].sum()

def sales_normalization(data,column):
    grouped_data = data.groupby(column)
    return grouped_data['sales_normalization'].sum()

def sales_by_time(data,duration):
    if duration == 'month':
        return data['Date'].dt.month
    if duration == 'day_of_week':
        return data['Date'].dt.day_of_week
    if duration == 'quarter':
        return data['Date'].dt.quarter
```
### Visualization
#### visualization/visualize.py
```
import seaborn as sb

import matplotlib.pyplot as plt

def bar_plot(data,xcolumn,ycolumn,huecolumn):
    plt.figure(figsize=(12, 8))
    sorted_data = data.sort_values([ycolumn], ascending=False)
    sb.barplot(x=xcolumn, y=ycolumn, hue=huecolumn, data=sorted_data)
    plt.title('State-wise Sales Analysis by Group')
    plt.show()

def line_plot(grouped_data):
    # Sorting time-of-day sales in descending order
    df_time_sales = grouped_data.sort_values(by="Sales", ascending=False)
    plt.figure(figsize=(10, 6))
    sb.lineplot(data=df_time_sales, x="Time", y="Sales", marker="o", linewidth=2, color="b")
    # sb.lineplot(x=df_time_sales.index, y=grouped_data.values)
    plt.title('Peak and Off-Peak Sales Analysis by Time of Day')
    plt.xlabel('Time of Day')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
```