
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
