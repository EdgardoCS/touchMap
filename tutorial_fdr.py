"""
author: EdgardoCS @FSU Jena
date: 02/04/2025
"""

# We start by creating a Pandas DataFrame of 1000 features. 990 of which (99%) will have their values generated
# from a Normal distribution with mean = 0, called a Null model
# The remaining 1% of the features will be generated from a Normal distribution mean = 3, called a Non-Null model.

import pandas as pd
import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

np.random.seed(42)

n_null = 9900
n_nonnull = 100

df = pd.DataFrame({
    'hypothesis': np.concatenate((
        ['null'] * n_null,
        ['non-null'] * n_nonnull,
    )),
    'feature': range(n_null + n_nonnull),
    'x': np.concatenate((
        norm.rvs(loc=0, scale=1, size=n_null),
        norm.rvs(loc=3, scale=1, size=n_nonnull),
    ))
})

def adjust_pvalues(p_values, method):
   return multipletests(p_values, method = method)[1]

df['p_value'] = 1 - norm.cdf(df['x'], loc = 0, scale = 1)

df['p_value_holm'] = adjust_pvalues(df['p_value'], 'holm')
df.sort_values('p_value_holm').head(10)

df['p_value_bh'] = adjust_pvalues(df['p_value'], 'fdr_bh')
df[['hypothesis', 'feature', 'x', 'p_value', 'p_value_holm', 'p_value_bh']].sort_values('p_value_bh').head(10)

df['is_p_value_holm_significant'] = df['p_value_holm'] <= 0.05

print(df.groupby(['hypothesis', 'is_p_value_holm_significant']).size())

df['is_p_value_bh_significant'] = df['p_value_bh'] <= 0.05

print(df.groupby(['hypothesis', 'is_p_value_bh_significant']).size())