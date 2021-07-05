
"""
Created on Wed June 30 9:00:00 2021

@author: Nuha and Rahaf
"""


import pandas as pd


# Drop columns with correlation above defined threshold
def drop_correlated_cols(df, cols_features, threshold):
    """
    Return a new dataframe, df, with columns correlated above threshold removed.
    The column removed is the right most one.
    """
    cols_to_remove = set()
    correlations_abs = df[cols_features].corr().abs()
    for i in range(len(correlations_abs.columns)):
        for j in range(i):
            if (correlations_abs.iloc[i, j] >= threshold) and (correlations_abs.columns[j] not in cols_to_remove):
                colname = correlations_abs.columns[i]
                if colname in df.columns:
                    cols_to_remove.add(colname)
                    df.drop(colname, axis=1, inplace=True)
    return df


# Load the entire data set
df = pd.read_excel('cyclicDisc.xlsx', na_values=['na', 'nan'])
print(df.shape)

dfEnthalpy = pd.DataFrame(df)
dfHeatCapacity = pd.DataFrame(df)
dfEntropy = pd.DataFrame(df)
dfBoiling = pd.DataFrame(df)
dfMelting = pd.DataFrame(df)


# Standard enthalpy dataframe
# Drop the columns
dfEnthalpy.drop(['NAME','SMILES','Boiling Point', 'Entropy' , 'Heat Capacity','Melting Point'], axis='columns', inplace=True)

# Drop rows with na
dfEnthalpy.dropna(subset = ["Standard Enthalpy"], inplace=True)
print(dfEnthalpy.shape)

# Drop columns with na
dfEnthalpy.dropna(axis='columns', inplace=True)

# Drop columns that have only a single value (variance = 0)
count_unique = dfEnthalpy.apply(pd.Series.nunique)
cols_to_drop = count_unique[count_unique == 1].index
dfEnthalpy.drop(columns=cols_to_drop, inplace=True)
print(dfEnthalpy.shape)

# Drop highly correlated features
cols_features = dfEnthalpy.columns[0:]
dfEnthalpy = drop_correlated_cols(dfEnthalpy, cols_features, 0.9)
print(dfEnthalpy.shape)


# Heat capacity dataframe
# Drop the columns
dfHeatCapacity.drop(['NAME','SMILES','Boiling Point', 'Entropy' , 'Standard Enthalpy','Melting Point'], axis='columns', inplace=True)

# Drop rows with na
dfHeatCapacity.dropna(subset = ["Heat Capacity"], inplace=True)
print(dfHeatCapacity.shape)

# Drop columns with na
dfHeatCapacity.dropna(axis='columns', inplace=True)

# Drop columns that have only a single value (variance = 0)
count_unique = dfHeatCapacity.apply(pd.Series.nunique)
cols_to_drop = count_unique[count_unique == 1].index
dfHeatCapacity.drop(columns=cols_to_drop, inplace=True)
print(dfHeatCapacity.shape)

# Drop highly correlated features
cols_features = dfHeatCapacity.columns[0:]
dfHeatCapacity = drop_correlated_cols(dfHeatCapacity, cols_features, 0.9)
print(dfHeatCapacity.shape)


# Entropy dataframe
# Drop the columns
dfEntropy.drop(['NAME','SMILES','Boiling Point', 'Heat Capacity' , 'Standard Enthalpy','Melting Point'], axis='columns', inplace=True)

# Drop rows with na
dfEntropy.dropna(subset = ["Entropy"], inplace=True)
print(dfEntropy.shape)

# Drop columns with na
dfEntropy.dropna(axis='columns', inplace=True)

# Drop columns that have only a single value (variance = 0)
count_unique = dfEntropy.apply(pd.Series.nunique)
cols_to_drop = count_unique[count_unique == 1].index
dfEntropy.drop(columns=cols_to_drop, inplace=True)
print(dfEntropy.shape)

# Drop highly correlated features
cols_features = dfEntropy.columns[0:]
dfEntropy = drop_correlated_cols(dfEntropy, cols_features, 0.9)
print(dfEntropy.shape)


# Boiling point dataframe
# Drop the columns
dfBoiling.drop(['NAME','SMILES','Melting Point', 'Standard Enthalpy' , 'Entropy' , 'Heat Capacity'], axis='columns', inplace=True)

# Drop rows with na
dfBoiling.dropna(subset = ["Boiling Point"], inplace=True)
print(dfBoiling.shape)

# Drop columns with na
dfBoiling.dropna(axis='columns', inplace=True)

# Drop columns that have only a single value (variance = 0)
count_unique = dfBoiling.apply(pd.Series.nunique)
cols_to_drop = count_unique[count_unique == 1].index
dfBoiling.drop(columns=cols_to_drop, inplace=True)
print(dfBoiling.shape)

# Drop highly correlated features
cols_features = dfBoiling.columns[0:]
dfBoiling = drop_correlated_cols(dfBoiling, cols_features, 0.9)
print(dfBoiling.shape)


# Melting point dataframe
# Drop the columns
dfMelting.drop(['NAME','SMILES','Boiling Point', 'Standard Enthalpy' , 'Entropy' , 'Heat Capacity'], axis='columns', inplace=True)

# Drop rows with na
dfMelting.dropna(subset = ["Melting Point"], inplace=True)
print(dfMelting.shape)

# Drop columns with na
dfMelting.dropna(axis='columns', inplace=True)

# Drop columns that have only a single value (variance = 0)
count_unique = dfMelting.apply(pd.Series.nunique)
cols_to_drop = count_unique[count_unique == 1].index
dfMelting.drop(columns=cols_to_drop, inplace=True)
print(dfMelting.shape)

# Drop highly correlated features
cols_features = dfMelting.columns[0:]
dfMelting = drop_correlated_cols(dfMelting, cols_features, 0.9)
print(dfMelting.shape)



# Save standard enthalpy dataset
dfEnthalpy.to_csv('CyclicEnthalpyProcessed.csv', index = False)

# Save heat capacity dataset
dfHeatCapacity.to_csv('CyclicHeatCapacityProcessed.csv', index = False)

# Save entropy dataset
dfEntropy.to_csv('CyclicEntropyProcessed.csv', index = False)

# Save melting point dataset
dfMelting.to_csv('CyclicMeltingPointProcessed.csv', index = False)

# Save boiling point dataset
dfBoiling.to_csv('CyclicBoilingPointProcessed.csv', index = False)