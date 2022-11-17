from math import sqrt, log1p
import numpy as np
from scipy.stats import norm

class GaussianIndependenceTest:
    def __init__(self, data_matrix, column_names):
        self.data_matrix = data_matrix
        self.inv_names = {name: i for i, name in enumerate(column_names)}
        self.n = data_matrix.shape[0]
        self.corr_matrix = np.corrcoef(data_matrix, rowvar=False)
        self.partial_corr_dict = {}

    def log_q1pm(self, x):
        """Returns log((1 + x) / (1 - x)) in a numerically stable way."""
        return log1p(2 * x / (1 - x))

    def partial_corr(self, i, j, K):
        K = list(K)
        if len(K) == 0:
            return self.corr_matrix[self.inv_names[i], self.inv_names[j]]
        else:
            # If the result is already cached
            idx = (i, j, frozenset(sorted(K)))
            if idx in self.partial_corr_dict:
                return self.partial_corr_dict[idx]

            h = K.pop()
            corr_i_h = self.partial_corr(i, h, K)
            corr_j_h = self.partial_corr(j, h, K)
            res = (self.partial_corr(i, j, K) - corr_i_h * corr_j_h) \
                / sqrt((1 - corr_i_h**2) * (1 - corr_j_h**2))
            # Cache the result
            #print("p-corr {} - {} | {} = {}".format(i, j, K + [h], res))
            self.partial_corr_dict[idx] = res
            return res


    def z_stat(self, i, j, K):
        r = self.partial_corr(i, j, K)
        return sqrt(self.n - len(K) - 3) * abs(0.5 * self.log_q1pm(r))

    def gaussian_indep_test(self, i, j, K):
        z = self.z_stat(i, j, K)
        return 2 * (1 - norm.cdf(z))
