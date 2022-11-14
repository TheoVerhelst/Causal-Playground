from gaussian import gaussian_indep_test

mu = [1, 3, 0.2]
sigma = [0.3, 0.1, 0.6]
beta_1 = 2.1
beta_2 = 3.1
N = 1000

U = np.empty(shape=(N, 3))
for i in range(3):
    U[:, i] = norm.rvs(loc = mu[i], scale = sigma[i], size = N)

X = np.empty(shape=(N, 3))
X[:, 0] = U[:, 0]
X[:, 1] = U[:, 1] + beta_1 * X[:, 0]
X[:, 2] = U[:, 2] + beta_2 * X[:, 0]

print("Expected: p=0, estimated: p={:.3f}".format(gaussian_indep_test(X, 1, 2, [])))
print("Expected: p=0, estimated: p={:.3f}".format(gaussian_indep_test(X, 0, 1, [])))
print("Expected: p=0, estimated: p={:.3f}".format(gaussian_indep_test(X, 0, 2, [])))
print("Expected: p>0, estimated: p={:.3f}".format(gaussian_indep_test(X, 1, 2, [0])))
print("Expected: p=0, estimated: p={:.3f}".format(gaussian_indep_test(X, 0, 1, [2])))
print("Expected: p=0, estimated: p={:.3f}".format(gaussian_indep_test(X, 0, 2, [1])))

