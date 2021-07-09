
"""
Created on Sun Jul 4 9:00:20 2021

@author: Nuha
"""
# Data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, DotProduct
from sklearn.metrics import r2_score


# Set seed
SEED = 123

data = pd.read_csv('CyclicBoilingPointProcessed.csv')

# Separate data into train and test
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Boiling Point']), # X: Remove the target label column and ID (not needed)
                                                    data['Boiling Point'], # y: Read the target label alone
                                                    test_size=0.25, # 25% testing, 75% training (we can change the size)
                                                random_state=SEED) # Fix the seed to the random generator

print(f"Shape of training features {X_train.shape}")
print(f"Shape of test features {X_test.shape}")

# Instantiate a Gaussian Process model
kernel = DotProduct() + WhiteKernel(noise_level=0.5)
gp = GaussianProcessRegressor(kernel=kernel)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X_train, y_train)

# Plot the function based on the R squared
predictions = gp.predict(X_test).flatten()
print(gp)
for p in predictions:
    print(p)  

r2 = r2_score(y_test, predictions)
plt.scatter(y_test,predictions)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')

plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, predictions, 1))(np.unique(y_test)))

plt.text(500, 100, 'R-squared = %0.2f' % r2)
plt.show()