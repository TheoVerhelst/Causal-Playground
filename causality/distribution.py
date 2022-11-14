import numpy as np
from causality.discrete_set import DiscreteSet

class IndependentDistribution:
    def __init__(self, dists: dict, seed=None):
        self.dists = dists
        self.generator = np.random.default_rng(seed)
     
    def rvs(self, size):
        """Generates a number of random samples of the distribution.
        
        :param size: The number of samples to generate
        :return: A dictionnary `{var: array}` where `var` is a
        `Variable`, and `array` is a numpy array of the sampled values.
        """
        return {var: dist.rvs(size=size, random_state=self.generator) \
                for var, dist in self.dists.items()}
    
    def pmf(self, set_: DiscreteSet):
        """Computes the probability of observing the given set of
        values.
        
        :param set_: A DiscreteSet denoting the values to consider. Must
        have the same dimensions as `self.dist.keys()`.
        :return: The probability of the set of values.
        """
        assert all(var in self.dists for var in set_.dimensions)
        proba_total = 0
        it = np.nditer(set_.values, flags=["multi_index"])
        for has_values in it:
            if has_values:
                proba_atom = 1
                for dim, val_i in enumerate(it.multi_index):
                    var = set_.dimensions[dim]
                    proba_atom *= self.dists[var].pmf(var.support[val_i])
                proba_total += proba_atom
        assert proba_total <= 1
        return proba_total
