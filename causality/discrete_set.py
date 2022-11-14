import numpy as np

class DiscreteSet:
    def __init__(self, dimensions, values: np.ndarray):
        """Contructor.
        
        Arguments:
        dimensions -- iterable of N variables denoting the dimensions
        values -- N-dimensional boolean array that says which
            combinations of values is included in the set.
        """
        self.dimensions = tuple(dimensions)
        self.values = values
    
    def tensor(self, other, axis):
        """Tensor product of both sets, where the sum is over axis."""
        i = self.dimensions.index(axis)
        j = other.dimensions.index(axis)
        assert self.dimensions[i] == other.dimensions[j], \
            "the dimension of the summed axis of self and other should match"
        dimensions = self.dimensions[:i] + self.dimensions[i + 1:] \
            + other.dimensions[:j] + other.dimensions[j + 1:]
        values = np.tensordot(self.values, other.values, (i,j))
        
        while True:
            try:
                for i, dim_i in enumerate(dimensions):
                    for j, dim_j in enumerate(dimensions[i + 1:], start=i + 1):
                        if dim_i == dim_j:
                            # Replace axes i and j by only one running along
                            # the diagonal of the sub-space (i, j). Numpy
                            # puts this new dimension at the end.
                            values = np.diagonal(values, axis1=i, axis2=j)
                            dimensions = dimensions[:i] \
                                    + dimensions[i + 1:j] \
                                    + dimensions[j + 1:] \
                                    + (dim_i,)
                            # Restart the outermost for loop if we found
                            # a duplicate dimension
                            raise StopIteration()
                break
            except StopIteration:
                continue
        
        return DiscreteSet(dimensions, values)
    
    def copy(self):
        return DiscreteSet(self.dimensions, self.values.copy())
    
    def _logical_op_helper(self, other, function):
        res = self.copy()
        res.match_to_broadcast(other)
        res.values = function(res.values, other.values)
        return res
    
    def __and__(self, other):
        return self._logical_op_helper(other, np.logical_and)
    
    def __or__(self, other):
        return self._logical_op_helper(other, np.logical_or)
    
    def __xor__(self, other):
        return self._logical_op_helper(other, np.logical_xor)
    
    def __sub__(self, other):
        return self._logical_op_helper(other, lambda a, b: a | ~b)
    
    def __invert__(self):
        res = self.copy()
        res.values = np.logical_not(res.values)
        return res
    
    def match_to_broadcast(self, other):
        """Moves and adds dimensions in self to be broadcastable with
        other. All the rightmost dimensions in self will be identical to
        the dimensions of other, and the dimensions unique to self will
        be the leftmost ones, which will be automatically broadcasted on
        other by numpy.
        
        Example input:
            self.dimensions  = (a, b, c)
            other.dimensions = (c, b, d)
        Result:
            self.dimensions  = (a, b, c, d)
        After this operation, any numpy operation such as self + other
        will broadcast other to add the dimensions (a,) to the left.
        Note that added dimensions in self (here, d) have a size of 1.
        """
        # Use list instead of tuple because dimensions will change here
        new_dims = list(self.dimensions)
        # Enumerate the dimensions in reverse order (i.e. starting from the right)
        for i_other, dim in enumerate(reversed(other.dimensions)):
            N = len(new_dims)
            try:
                i_self = new_dims.index(dim)
                assert i_self >= i_other
                # Swap axes in self
                new_dims[i_self], new_dims[N - i_other] = new_dims[N - i_other], new_dims[i_self]
                self.values = np.swapaxes(self.values, N - i_other, i_self)
            except ValueError:
                # Add an axis in self
                new_dims.insert(N - i_other, dim)
                self.values = np.expand_dims(self.values, N - i_other)
        self.dimensions = tuple(new_dims)
