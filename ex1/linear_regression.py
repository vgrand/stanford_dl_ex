import numpy as np

def partial_cost(theta, x, y):
	(n_rows, n_obs) = x.shape

	pc = np.zeros(n_obs)

	for i in range(n_obs):
		h_th = 0.0
		for j in range(n_rows):
			h_th += theta[j]*x[j,i]
		pc[i] = h_th - y[i]
	return pc

# Arguments:
# theta - A vector containing the parameter values to optimize.
# X - The examples stored in a matrix.
#    X(i,j) is the i'th coordinate of the j'th example.
# y - The target value for each example.  y(j) is the target for example j.

# Output
# f - cost function evaluated in theta

def linear_regression(theta, x, y):

	f = 0.0

	(n_rows, n_obs) = x.shape

	pc = partial_cost(theta, x, y)	

	for i in range(n_obs):
		f += pc[i]**2

	return f

# Arguments:
# theta - A vector containing the parameter values to optimize.
# X - The examples stored in a matrix.
#    X(i,j) is the i'th coordinate of the j'th example.
# y - The target value for each example.  y(j) is the target for example j.

# Output
# g - gradient of the cost function
def linear_regression_der(theta, x, y):
	(n_rows, n_obs) = x.shape
	g = np.zeros(theta.shape)

	pc = partial_cost(theta, x, y)

	for j in range(n_rows):
		for i in range(n_obs):
			g[j] += x[j,i]*pc[i]

	return g


def linear_regression_vec(theta, x, y):
	return 0.5 * np.sum( (theta.transpose().dot(x) - y)**2 )

def linear_regression_der_vec(theta, x, y):
	return np.sum( x * (theta.transpose().dot(x) - y) , axis = 1)