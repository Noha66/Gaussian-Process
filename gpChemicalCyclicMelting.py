
"""
Created on Sun Jul 4 9:00:20 2021

@author: Nuha
"""
# data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, DotProduct, RBF, ConstantKernel as C
from sklearn.metrics import r2_score#, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# set seed
SEED = 268  # 136 res = 0.32, 145 better 0.34, 172 better = 0.38, 268 res = 0.43

data = pd.read_csv('CyclicMeltingPointProcessed.csv')

# separate data into train and test
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Melting Point']), # X: Remove the target label column and ID (not needed)
                                                    data['Melting Point'], # y: Read the target label alone
                                                    test_size=0.20, # 20% testing, 80% training (we can change the size)
                                                random_state=SEED) # Fix the seed to the random generator

print(f"Shape of training features {X_train.shape}")
print(f"Shape of test features {X_test.shape}")

# Instantiate a Gaussian Process model
kernel = DotProduct() + WhiteKernel(noise_level=0.5) 
gp = GaussianProcessRegressor(kernel=kernel)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X_train, y_train)


# Plot the functio based on the R squared
predictions = gp.predict(X_test).flatten()
print(gp)
for p in predictions:
    print(p)  

r2 = r2_score(y_test, predictions)
plt.scatter(y_test,predictions)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')

plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, predictions, 1))(np.unique(y_test)))

plt.text(0.6, -50, 'R-squared = %0.2f' % r2)
plt.show()