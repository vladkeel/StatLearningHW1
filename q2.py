import os
import pandas as pd
from statsmodels.formula.api  import ols
import matplotlib.pyplot as plt
import numpy as np
# Part 1:
df = pd.read_csv("parkinsons_updrs_data.csv")
# Part 2:
print(df.describe())

# Part 3:
print(df.columns)
used_variables = ['motor_UPDRS', 'Jitter.Per', 'Jitter.Abs', 'Shimmer', 'Shimmer.dB', 'NHR', 'HNR']
part_df = df[used_variables]
part_df = part_df.rename(index=str, columns={'Jitter.Per' : 'Jitter_Per', 'Jitter.Abs' : 'Jitter_Abs', 'Shimmer.dB' : 'Shimmer_dB'})
#grr = pd.plotting.scatter_matrix(part_df, figsize=(10, 10))
#plt.show()

# Part 4:
def rse(X, y):
    o = np.ones((np.size(X, 0), 1))
    Xwone = np.append(o, X, axis=1)
    xtx = np.matmul(Xwone.T, Xwone)**-1
    xty = np.matmul(Xwone.T, y.reshape(y.size,1))
    return np.matmul(xtx, xty)


dfx = df[['Jitter.Per', 'Jitter.Abs', 'Shimmer', 'Shimmer.dB', 'NHR', 'HNR']]
dfy = df['motor_UPDRS']
print(rse(np.array(dfx), np.array(dfy)))

# Part 5:

est = ols(formula=r' motor_UPDRS ~  Jitter_Per + Jitter_Abs + Shimmer + Shimmer_dB + NHR + HNR', data=part_df).fit()
print(est.summary())

