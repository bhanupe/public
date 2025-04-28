import pandas as pd
# Set the display.max_columns option to None
from analytical_processors.visualization.visualize import heat_map_missing, univariate_analysis, bivariate_analysis, \
    multivariate_analysis, box_plot
from analytical_processors.wrangling.insights import explain
from analytical_processors.wrangling.prepare import encode, add_zscores

data = pd.read_csv("lending_club/data/loan_data.csv")
explain(data)
heat_map_missing(data)

data_encoded = encode(data, int)
explain(data_encoded)
heat_map_missing(data_encoded)

univariate_analysis(data)

bivariate_analysis(data, 'not.fully.paid')

numeric_data = data.drop(columns='purpose')
multivariate_analysis(numeric_data, 'not.fully.paid')

z_scores, data_with_zscores = add_zscores(data)
explain(data_with_zscores)

box_plot(z_scores)
