#Import modules
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.signal import (argrelmin, argrelmax)
#Generate data
np.random.seed(0)
X = np.linspace(0, 10, 100)[:, np.newaxis]
y = np.sin(X) + np.random.normal(0, 0.2, X.shape)

#Define the type of variables
var_type = 'c'

#Define the type of regression estimator
reg_type = 'll'

#Train the model
model = KernelReg(y, X, var_type, reg_type=reg_type,bw='cv_ls')

X1 = np.linspace(0, 10, 50)[:, np.newaxis]
y_pred, _ = model.fit(X1)

# 请多平滑后的高低点
max_index = argrelmax(y_pred)[0]
min_index = argrelmin(y_pred)[0]

#Plot the results
plt.scatter(X, y, color='blue', label='data')
plt.plot(X1, y_pred, color='red', label='kernel regression')
plt.scatter(X1[max_index], y_pred[max_index], color='red', label='max', linewidths=5)
plt.legend()
plt.show()