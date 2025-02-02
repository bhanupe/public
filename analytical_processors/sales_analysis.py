# %%
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# %%
file_path = '/Users/nagaswethaakula/pythonCourse/AppliedDatascienceAssignment/AusApparalSales4thQrt2020.csv'
data = pd.read_csv(file_path)
data.head(100)

# %%
print(f'Shape : \t{data.shape}')
print(f'Row Labels : \t{data.index}')
print(f'Column Names : \n{data.columns}')
print(f'DataType : \n{data.dtypes}')

# %% [markdown]
#
# # a. no of missing values in each column
# # Inspect the data manually to identify missing or incorrect
# information using the functions isna() and notna().

# %%
data.isna().sum(axis = 0)

# %%
data.head()

# %%
data.Sales.isna()

# %%
data.info()

# %%
data.iloc[0]

# %%
data.loc[0]

# %%
data.notna().sum(axis = 0)

# %%
data.head(2)

# %%
data.Sales.unique

# %%
data.sort_values(['State'], ascending= True).head(20)
data.State.unique()

# %%
data.head(20)

# #  b. Based on your knowledge of data analytics, include your
# recommendations for treating missing and incorrect data (dropping the
# null values or filling them).

# TBD C. Choose a suitable data wrangling technique—either data standardization
# or normalization. Execute the preferred normalization method and
# present the resulting data. (Normalization is the preferred approach for this
# problem.)  Normalization (Min-Max Scaling)	When data needs to be in a fixed range (0 to 1)	Xnormalized=Xmax−XminX−Xmin

scaler = MinMaxScaler()
# Apply normalization to 'Sales' column
data['sales_normalization'] = scaler.fit_transform(data[['Sales']])
print(data)
print(data.loc[data['Sales'] == 5000])
print(data[['Unit','Sales','sales_normalization']].agg(['min','max']))

# %% [markdown]
# # convert Date column from object to Date type using to_datetime function
# %%
data.Date

# %%
pd.to_datetime(data.Date)

# %%
data.info()

# %% [markdown]
# Assign it to Data after converting in to Date Type

# %%
data.Date = pd.to_datetime(data.Date)

# %% [markdown]
# After assigning Date column is updated to datetime64

# %%
data.info()

# %%
import datetime as dt
data.Date.dt.year

# %%
data.Date.dt.month

# %% a. Perform descriptive statistical analysis on the data in the Sales and Unit
# columns. Utilize techniques such as mean, median, mode, and standard
# deviation for this analysis.
summary = data.describe()
summary

# %%
data.select_dtypes(include = 'int' ).describe()

# %%
data.select_dtypes(include = 'object' ).describe()

# %%
data.select_dtypes(include = 'object').mode()

# %%
data.head(2)

# %%
data.mode(numeric_only= True)

# # Perform descriptive statistical analysis on the data in the Sales and Unit
# #columns. Utilize techniques such as mean, median, mode, and standard
# #deviation for this analysis.


groupedValues = data.groupby('State')

# %%
print(groupedValues[['Sales']].mean())

# %%
print(groupedValues.head(2))

# %%
print(groupedValues.mean(numeric_only=True))

# %%
print(groupedValues.agg(['min', 'max']))

# %%
print(groupedValues[['Unit','Sales']].agg(['mean','std']))


groupedByColumnUnit = data.groupby('Unit')
print(groupedByColumnUnit)

groupedByColumnGroup = data.groupby('Group',)
print(groupedByColumnGroup)


groupedByColumnGroupState = data.groupby('Group')
print(groupedByColumnGroupState)

groupedByColumnDate = data.groupby('Date')
print(groupedByColumnDate)

groupedByColumnTime = data.groupby('Time')
print(groupedByColumnTime)

# %% d. Share your insights regarding the application of the GroupBy() function for
# either data chunking or merging, and offer a recommendation based on
# your analysis.

# Grouping the data by 'State' and 'Time' to analyze total sales and units sold
data_grouped = data.groupby(['State', 'Time']).agg({'Sales': 'sum', 'Unit': 'sum'}).reset_index()
print(data_grouped)

# Grouping the data by 'State' and 'Time' to analyze total sales and units sold
data_grouped_group = data.groupby(['State', 'Time', 'Group']).agg({'Sales': 'sum', 'Unit': 'sum'}).reset_index()
print(data_grouped_group)

sorted_group = data_grouped_group.sort_values(by='Sales',ascending=False)
print(sorted_group)
print(sorted_group)

# Applying standard deviation to the 'Sales' column and creating a new column
data["Sales_Std_Dev"] = data.groupby(["State", "Time"])["Sales"].transform("std")
print(data)

# %% [markdown]
# # Using the GroupBy() function to analyze aggregated statistics such as total sales by state

# %%
salesByState = groupedValues['Sales'].sum()
salesByState

# %% [markdown]
# #### Identify the group with the highest sales and the group with the lowest sales based on the data provided.

# %%
groupedByColumnGroup = data.groupby('Group')

# %% [markdown]
# # Sales by group

# %%
salesByGroup = groupedByColumnGroup['Sales'].sum()
salesByGroup

# %% [markdown]
# ### Sorting for highest and lowest sales

# %%
highest_sales_group = salesByGroup.idxmax()
highest_sales_group

# %%
lowest_sales_group = salesByGroup.idxmin()
lowest_sales_group

# %% [markdown]
# # Sales by state

# %%
salesByState = groupedValues['Sales'].sum()
salesByState

# %% [markdown]
# ### Sorting for highest and lowest sales by State

# %%
highest_sales_state = salesByState.idxmax()
highest_sales_state

# %%
lowest_sales_state = salesByState.idxmin()
lowest_sales_state

# %%
groupedByColumnGroup[['Unit','Sales']].agg(['min','max'])

# %%
groupedByColumnGroup[['Unit','Sales']].agg(['mean','std'])

# %%
groupedByColumnDate = data.groupby('Date')

# %%
groupedByColumnDate[['Unit','Sales']].agg(['min','max'])

# %% [markdown]
# # Weekly, Monthly, and Quarterly Reports
# ### d. Generate weekly, monthly, and quarterly reports to document and present the results of the analysis conducted.

# %%
data.Date

# %% [markdown]
# # monthly Report

# %%
dataPerMonth= data['Date'].dt.month
dataPerMonth

# %%

monthySales = data.groupby(dataPerMonth)['Sales'].sum()
monthySales

# %%
dataPerWeek= data['Date'].dt.day_of_week
dataPerWeek

# %% [markdown]
# ## Weekly report

# %%

weeklySales = data.groupby(dataPerWeek)['Sales'].sum()
weeklySales

# %% [markdown]
# ## quarterly reports

# %%
dataPerQuarter = data['Date'].dt.quarter
dataPerQuarter

# %%
quarterReport = data.groupby(dataPerQuarter)['Sales'].sum()
quarterReport

# %% [markdown]
# # Data Analysis

# %%
# Descriptive statistics for Sales and Unit
salesDesc = data['Sales'].describe()
salesDesc

# %%
unitDesc = data['Unit'].describe()
unitDesc

# %%
# Mode
salesMode = data['Sales'].mode()
salesMode

# %%
unitMode = data['Unit'].mode()
unitMode

# %%

# Standard deviation
sales_std = data['Sales'].std()
sales_std

# %%
unit_std = data['Unit'].std()
unit_std.__round__(2)

# %% [markdown]
# # Data Visualization



plt.figure(figsize=(12, 8))
sb.barplot(x='State', y='Sales', hue='Group', data=data)
plt.title('State-wise Sales Analysis by Group')
plt.show()


# %% [markdown]
# # Group-wise sales by state

# %%

group_sales_by_state = data.groupby(['State', 'Group'])['Sales'].sum().unstack()
group_sales_by_state

# %%

plt.figure(figsize=(20, 10))
sb.heatmap(group_sales_by_state, annot=True, cmap='coolwarm', fmt='.2g',linewidths = '2')
plt.title('Group-wise Sales by State')
plt.show()

# %% [markdown]
# # Time-of-the-day analysis: Identify peak and off-peak sales periods
#

# %%

salesByTime= data.groupby('Time')['Sales'].sum()
salesByTime


# %%
plt.figure(figsize=(8, 5))
sb.lineplot(x=salesByTime.index, y=salesByTime.values)
plt.title('Sales by Time of the Day')
plt.xlabel('Time of Day')
plt.ylabel('Total Sales')
plt.show()

# %% [markdown]
# # VIC state has highest sales and WA state has lowest sales
# # MEN group has highest sales and Seniors group has lowest sales
# # Peak sales hour is in Morning time of the day.


