import os
import pandas as pd
from statsmodels.formula.api  import ols
import matplotlib

# Part 1:
df = pd.read_csv("parkinsons_updrs_data.csv")
# Part 2:
print(df.describe())

# Part 3:
print(df.columns)
used_variables = ['motor_UPDRS', 'Jitter.Per', 'Jitter.Abs', 'Shimmer', 'Shimmer.dB', 'NHR', 'HNR']
part_df = df[used_variables]
pd.plotting.scatter_matrix(part_df, diagonal='kde', figsize=(10, 10))
