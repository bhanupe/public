import pandas as pd

from analytical_processors.analysis.data_analysis import sales, sales_normalization, sales_by_time
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



    except Exception as e:
        print(f"❌ Fatal error: {e}")