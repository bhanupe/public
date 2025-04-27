import pandas as pd
# Set the display.max_columns option to None
from analytical_processors.wrangling.insights import explain
from analytical_processors.wrangling.prepare import encode

data = pd.read_csv("../data/loan_data.csv")
explain(data)
data_encoded = encode(data, int)
explain(data_encoded)

