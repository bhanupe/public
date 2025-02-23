import seaborn as sb

import matplotlib.pyplot as plt


def bar_plot(data, xcolumn, ycolumn, huecolumn, title):
    plt.figure(figsize=(12, 8))
    sorted_data = data.sort_values([ycolumn], ascending=False)
    sb.barplot(x=xcolumn, y=ycolumn, hue=huecolumn, data=sorted_data)
    plt.title(title)
    plt.show()


def line_plot(grouped_data):
    # Sorting time-of-day sales in descending order
    df_time_sales = grouped_data.sort_values(by="Sales", ascending=False)
    plt.figure(figsize=(10, 6))
    sb.lineplot(data=df_time_sales, x="Time", y="Sales", marker="o", linewidth=2, color="b")
    # sb.lineplot(x=df_time_sales.index, y=grouped_data.values)
    plt.title('Time-of-the-day analysis')
    plt.xlabel('Time of Day')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


def heat_map(correlation_matrix):
    plt.figure(figsize=(20, 10))
    sb.heatmap(correlation_matrix, annot=True, cmap='BuGn', fmt='.2g', linewidths='2')
    plt.title('Correlation matrix')
    plt.show()

def hist_plot(data):
    plt.figure(figsize=(20, 10))
    sb.histplot(data, kde=True, bins=7,color="teal")
    plt.title('Distribution Plot')
    plt.ylabel("Employee Count")
    plt.show()


def  count_plot(data,xcolumnValue, hueValue,title, xlabel,ylabel,statValue) :
    sb.countplot(data, x=xcolumnValue, hue=hueValue,stat = statValue)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

