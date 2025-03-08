import pandas as pd

from analysis.data_analysis import group_data, group_data_by_time
from visualization.visualize import bar_plot, line_plot
from wrangling.insights import explain, group_by_features
from wrangling.normalization import min_max_normalization
from wrangling.prepare import fill_null, convert_data_type

# Main Execution
if __name__ == "__main__":
    try:
        file_path = '../data/AusApparalSales4thQrt2020.csv'
        data = pd.read_csv(file_path)
        explain(data)
        # fill_null(data,'Time','Morning')
        # fill_null(data, 'Sales', 20000)
        explain(data)
        data = convert_data_type(data, data.Date, 'datetime')
        print('After converting datatime datatype')
        print(data.info())
        min_max_normalization(data, 'Sales', 'sales_normalization')
        print('after adding sales_normalization')
        print(data[['Unit', 'Sales', 'sales_normalization']].agg(['min', 'max']))
        group_by_features(data)
        grouped_data_by_state = group_data(data, 'State', 'Sales')
        print(f"Maximum Sales in State={grouped_data_by_state.idxmax()}")
        print(f"Minimum Sales in State={grouped_data_by_state.idxmin()}")
        grouped_data_by_group = group_data(data, 'Group', 'Sales')
        print(f"Maximum Sales by Group={grouped_data_by_group.idxmax()}")
        print(f"Minimum Sales by Group={grouped_data_by_group.idxmin()}")

        grouped_data_by_state_normalization = group_data(data, 'State', 'sales_normalization')
        print(f"Maximum Sales using normalization in State={grouped_data_by_state_normalization.idxmax()}")
        print(f"Minimum Sales using normalization in State={grouped_data_by_state_normalization.idxmin()}")
        grouped_data_by_group_normalization = group_data(data, 'Group', 'sales_normalization')
        print(f"Maximum Sales using normalization by Group={grouped_data_by_group_normalization.idxmax()}")
        print(f"Minimum Sales using normalization by Group={grouped_data_by_group_normalization.idxmin()}")

        monthly_report = group_data_by_time(data, 'month')
        print(f"Monthly Sales={data.groupby(monthly_report)['Sales'].sum()}")
        weekly_report = group_data_by_time(data, 'day_of_week')
        print(f"Weekly Sales={data.groupby(weekly_report)['Sales'].sum()}")
        quarterly_report = group_data_by_time(data, 'quarter')
        print(f"Quarterly Sales={data.groupby(quarterly_report)['Sales'].sum()}")

        monthly_report_sales_normalization = group_data_by_time(data, 'month')
        print(f"Monthly sales_normalization={data.groupby(monthly_report)['sales_normalization'].sum()}")
        weekly_report_sales_normalization = group_data_by_time(data, 'day_of_week')
        print(f"Weekly sales_normalization={data.groupby(weekly_report)['sales_normalization'].sum()}")
        quarterly_report_sales_normalization = group_data_by_time(data, 'quarter')
        print(f"Quarterly sales_normalization={data.groupby(quarterly_report)['sales_normalization'].sum()}")

        bar_plot(data, 'State', 'Sales', 'Group', 'State-wise sales analysis')
        bar_plot(data, 'Group', 'Sales', 'State', 'Group-wise sales analysis')
        sorted_data = data.sort_values(['Time'], ascending=False)
        line_plot(sorted_data.groupby('Time')['Sales'].sum().reset_index())
    except Exception as e:
        print(f"❌ Fatal error: {e}")
