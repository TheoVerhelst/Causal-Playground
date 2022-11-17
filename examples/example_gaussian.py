import numpy as np
from scipy.stats import norm
from causality.gaussian import GaussianIndependenceTest

# Define the parameters of three independent Gaussian distributions
mu = [1, 3, 0.2]
sigma = [0.3, 0.1, 0.6]
beta_1 = 2.1
beta_2 = 3.1
N = 1000

# Generate independent samples from the distributions
U = np.empty(shape=(N, 3))
for i in range(3):
    U[:, i] = norm.rvs(loc = mu[i], scale = sigma[i], size = N)

# Define X, Y and Z according to the causal graph Z <- X -> Y
X = np.empty(shape=(N, 3))
X[:, 0] = U[:, 0]
X[:, 1] = U[:, 1] + beta_1 * X[:, 0]
X[:, 2] = U[:, 2] + beta_2 * X[:, 0]
test = GaussianIndependenceTest(X, ["X", "Y", "Z"])

print("Expected: p=0, estimated: p={:.3f}".format(test.gaussian_indep_test("Y", "Z", [])))
print("Expected: p=0, estimated: p={:.3f}".format(test.gaussian_indep_test("X", "Y", [])))
print("Expected: p=0, estimated: p={:.3f}".format(test.gaussian_indep_test("X", "Z", [])))
print("Expected: p>0, estimated: p={:.3f}".format(test.gaussian_indep_test("Y", "Z", ["X"])))
print("Expected: p=0, estimated: p={:.3f}".format(test.gaussian_indep_test("X", "Y", ["Z"])))
print("Expected: p=0, estimated: p={:.3f}".format(test.gaussian_indep_test("X", "Z", ["Y"])))

