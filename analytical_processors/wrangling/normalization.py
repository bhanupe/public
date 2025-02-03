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