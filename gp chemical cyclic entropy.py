# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:31:04 2021

@author: Rahaf
"""

# data analysis
import pandas as pd

# display all columns
pd.set_option('max.columns', None)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, Sum, Matern, WhiteKernel, DotProduct


# set seed
SEED = 123

# read data
data = pd.read_csv('CyclicEntropyProcessed.csv')

# separate data into train and test
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Entropy']), # X: Remove the target label column and ID (not needed)
                                                    data['Entropy'], # y: Read the target label alone
                                                    test_size=0.3, # 30% testing, 70% training (we can change the size)
                                                random_state=SEED) # Fix the seed to the random generator
# str.encode().decode()
print(f"Shape of training features {X_train.shape}")
print(f"Shape of test features {X_test.shape}")

# Instantiate a Gaussian Process model

kernel = Sum(C(2), RBF())
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)

# kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
# gp = GaussianProcessRegressor(kernel=kernel,random_state=0)

# kernel = DotProduct() + WhiteKernel(noise_level=0.5)
# gp = GaussianProcessRegressor(kernel=kernel,random_state=0)

# kernel = RBF() + C(constant_value=2)
# gp = GaussianProcessRegressor(kernel=kernel, alpha=5,random_state=0)

# kernel = 1.0 * RBF(1.0)
# gp = GaussianProcessRegressor(kernel=kernel,random_state=0)

# kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
# gp = GaussianProcessRegressor(kernel=kernel,random_state=0)


# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X_train, y_train)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(X_test, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
predictions = gp.predict(X_test).flatten()
print(gp)
for p in predictions:
    print(p)  

r2 = r2_score(y_test, predictions)
plt.scatter(y_test,predictions)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')

plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, predictions, 1))(np.unique(y_test)))

plt.text(0.6, 0.5, 'R-squared = %0.2f' % r2)
plt.show()
