import pandas as pd
# Set the display.max_columns option to None
from analytical_processors.wrangling.insights import explain

data = pd.read_csv("../data/loan_data.csv")
explain(data)
data_not_delinq=data.filter(items=['delinq.2yrs'])
explain(data_not_delinq)
