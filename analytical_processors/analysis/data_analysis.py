# Identify the group with the highest sales and the group with the lowest
# sales based on the data provided.

# Group data column to find the influence on a target. e.g. sales by state or member group (kids, seniors etc)
def group_data(data, column, target):
    grouped_data = data.groupby(column)
    return grouped_data[target].sum()


def group_data_by_time(data, duration, date_column='Date'):
    if duration == 'month':
        return data[date_column].dt.month
    if duration == 'day_of_week':
        return data[date_column].dt.day_of_week
    if duration == 'quarter':
        return data[date_column].dt.quarter
