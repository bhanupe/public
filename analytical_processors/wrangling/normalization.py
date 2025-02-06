from sklearn.preprocessing import MinMaxScaler

# TBD C. Choose a suitable data wrangling technique—either data standardization
# or normalization. Execute the preferred normalization method and
# present the resulting data. (Normalization is the preferred approach for this
# problem.)  Normalization (Min-Max Scaling)	When data needs to be in a fixed range (0 to 1)	Xnormalized=Xmax−XminX−Xmin
def min_max_normalization(data,column_name,new_column_name_to_be_added):
    scaler = MinMaxScaler()
    # Apply normalization to 'Sales' column
    data[new_column_name_to_be_added] = scaler.fit_transform(data[[column_name]])
    return data