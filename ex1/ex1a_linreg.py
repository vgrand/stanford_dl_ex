import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

import time
import sys

import linear_regression

# load data from the csv housing prices
data = np.loadtxt(open('housing.csv', 'rb'), delimiter = ',')
# put examples in columns
data = data.transpose()

# add a row of 1s as an additional intercept feature
data = np.concatenate( (np.ones((1,data.shape[1])), data), axis = 0 )
(nb_rows, nb_obs) = data.shape

# shuffle data
data = data[:, np.random.permutation(nb_obs)]

# split into train and test sets
train_X = data[0:nb_rows-1,0:400]
train_y = data[nb_rows-1,0:400]

test_X = data[0:nb_rows-1,400:]
test_y = data[nb_rows-1,400:]

(n,m) = train_X.shape

# init vector to random values
theta = np.random.rand(n,1)

vec = True

t0 = time.time()
if vec:
	lin_reg = linear_regression.linear_regression_vec
	lin_reg_grad = linear_regression.linear_regression_der_vec
else:
	lin_reg = linear_regression.linear_regression
	lin_reg_grad = linear_regression.linear_regression_der

res = scipy.optimize.minimize(
	fun = lin_reg, x0 = theta,
	args = (train_X, train_y), jac = lin_reg_grad,
	method = 'BFGS', options = {'maxiter' : 200})

theta = res.x
t1 = time.time()

# No convergence
if not(res.success):
	print('Did not converge !')
	sys.exit(1)

print('Took : '+str(t1-t0)+'secs')
#print theta
#print theta.shape

# print rms from training set
actual_prices = train_y
predicted_prices = theta.transpose().dot(train_X)

train_rms = np.sqrt( np.mean( ( actual_prices - predicted_prices )**2 ) )
print('RMS training error : ' + str(train_rms))

# print rms from testing set
actual_prices = test_y
predicted_prices = theta.transpose().dot(test_X)

test_rms = np.sqrt( np.mean( (actual_prices - predicted_prices)**2 ) )
print('RMS testing error : ' + str(test_rms))

# plot predictions from testing set
idx = np.argsort(actual_prices)

actual_prices_sorted = actual_prices[idx]
predicted_prices_sorted = predicted_prices[idx]
house_no = range(actual_prices.size)
plot_actual_prices, = plt.plot(house_no, actual_prices_sorted, 'rx')
plot_predicted_prices, = plt.plot(house_no, predicted_prices_sorted, 'bx')
plt.legend([plot_actual_prices, plot_predicted_prices],['Actual prices', 'Predicted prices'])
plt.xlabel('House #')
plt.ylabel('House prices ($1000s)')
plt.show()