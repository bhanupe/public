import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

show = True


def save_show(plot, name, identifier=""):
    plot.savefig(f'{name}_{identifier}.png')
    if show:
        plot.show()


def bar_plot(data, xcolumn, ycolumn, huecolumn, title, identifier=""):
    plt.figure(figsize=(12, 8))
    sorted_data = data.sort_values([ycolumn], ascending=False)
    sb.barplot(x=xcolumn, y=ycolumn, hue=huecolumn, data=sorted_data)
    plt.title(title)
    method_name = __name__
    save_show(plt, f'bp{method_name}', identifier)


def line_plot(grouped_data, identifier=""):
    print(__name__)
    # Sorting time-of-day sales in descending order
    df_time_sales = grouped_data.sort_values(by="Sales", ascending=False)
    plt.figure(figsize=(10, 6))
    sb.lineplot(data=df_time_sales, x="Time", y="Sales", marker="o", linewidth=2, color="b")
    # sb.lineplot(x=df_time_sales.index, y=grouped_data.values)
    plt.title('Time-of-the-day analysis')
    plt.xlabel('Time of Day')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.grid(True, identifier)

    method_name = __name__
    save_show(plt, f'lp{method_name}', identifier)


def multivariate_analysis(data, target, identifier=""):
    plt.figure(figsize=(20, 10))
    sb.heatmap(data.corr(), annot=True, cmap='BuGn', fmt='.2g', linewidths='2')
    plt.title('Correlation matrix')
    save_show(plt, f'cm{__name__}', identifier)

    # Optional: Pairplot
    sb.pairplot(data, hue=target)
    plt.title('Pair plot')
    save_show(plt, f'pp{__name__}', identifier)


def heat_map_missing(data, identifier=""):
    plt.figure(figsize=(20, 10))
    null_counts = data.isnull()

    sb.heatmap(null_counts, cbar=False, cmap='viridis')
    plt.title('Missing Values Heat Map')
    save_show(plt, f'mv{__name__}', identifier)

    # Plot the bar chart
    null_counts.sum().plot(kind='bar', title='Null Values in Each Column', figsize=(20, 10))
    # Add labels and rotate x-axis labels for better readability
    plt.xlabel('Columns')
    plt.ylabel('Number of Null Values')
    plt.xticks(rotation=45)

    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    save_show(plt, f'nv{__name__}', identifier)


def hist_plot(data, identifier=""):
    plt.figure(figsize=(20, 10))
    sb.histplot(data, kde=True, bins=7, color="teal")
    plt.title('Distribution Plot')
    plt.ylabel("Employee Count")
    save_show(plt, f'hp{data.name}_{__name__}', identifier)


def count_plot(data, xcolumnValue, hueValue, title, xlabel, ylabel, statValue, identifier=""):
    sb.countplot(data, x=xcolumnValue, hue=hueValue, stat=statValue)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_show(plt, f'cp{xcolumnValue}_{hueValue}_{__name__}', identifier)


def univariate_analysis(data, identifier=""):
    # Same as describe
    columns = data.columns
    # for column in columns:
    #     printline()
    #     print(f'Analyzing {column}: \n{data[column].describe()}')
    # Numeric Features
    # columns = data.columns
    for column in columns:
        if type(column) == object:
            # Categorical Feature
            sb.countplot(y=column, data=data)
            plt.title(f'{column} Distribution')
            save_show(plt, f'uao{column}_{__name__}')
        else:
            sb.histplot(data[column], kde=True, bins=30)
            plt.title(f'{column} Score Distribution')
            save_show(plt, f'uan{column}_{__name__}', identifier)


def bivariate_analysis(data, target, identifier=""):
    columns = data.columns
    for column in columns:
        if type(column) == object:
            # Categorical Feature
            purpose_default = pd.crosstab(data[column], data[target])
            purpose_default.div(purpose_default.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
            plt.title(f'{target} by {column}')
            save_show(plt, f'bao{column}_{__name__}', identifier)
        else:
            sb.boxplot(x=target, y=column, data=data)
            plt.title(f'{column} vs {target}')
            save_show(plt, f'ban{column}_{__name__}', identifier)


def box_plot(data, identifier=""):
    plt.figure(figsize=(10, 6))
    sb.boxplot(data=data)
    plt.title('Box Plot of Z-Scores')
    plt.xlabel('Columns')
    plt.ylabel('Z-Score')
    save_show(plt, f'zsbp{__name__}', identifier)
