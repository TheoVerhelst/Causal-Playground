from operator import mul
from typing import Sequence
from functools import reduce
import numpy as np
from causality.variable import Variable


class DiscreteSet:
    def __init__(self, dimensions: Sequence[Variable], values: np.ndarray):
        """Contructor.
        
        :param dimensions: iterable of N variables denoting the dimensions
        :param values: N-dimensional boolean array indicating which
            combinations of values are included in the set.
        """
        self.dimensions = tuple(dimensions)
        self.values = values
    
    def tensor(self, other, axis: Variable):
        """Tensor product of both sets, where the sum is over `axis`.
        Dimensions common to `self` and `other` are collapsed into one.
        """
        
        common_dimensions = self._match_to_tensor(other, axis)
        common_size = reduce(mul, (len(dim.support) for dim in common_dimensions), 1)
        self_values = self.values.reshape((len(axis.support), common_size, -1))
        other_values = other.values.reshape((len(axis.support), common_size, -1))

        values = np.einsum("ijk,ijl->jkl", self_values, other_values)
        dimensions = tuple(common_dimensions) \
                + self.dimensions[len(common_dimensions) + 1:] \
                + other.dimensions[len(common_dimensions) + 1:]
        values = values.reshape(tuple(len(dim.support) for dim in dimensions))
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
    
    def _match_to_tensor(self, other, axis: Variable):
        """
        Example input:
        ```
            axis = c
            self.dimensions = (a, d, c, b)
            other.dimensions = (e, f, c, b)
        ```
        Result:
        ```
            self.dimensions = (c, b, a, d)
            other.dimensions = (c, b, e, f)
        ```
        """
        self._swap_axes(0, self.dimensions.index(axis))
        other._swap_axes(0, other.dimensions.index(axis))
        common_dimensions = []
        for i_self, dim in enumerate(self.dimensions[1:], start=1):
            if dim in other.dimensions:
                i_other = other.dimensions.index(dim)
                self._swap_axes(len(common_dimensions) + 1, i_self)
                other._swap_axes(len(common_dimensions) + 1, i_other)
                common_dimensions.append(dim)
        return common_dimensions
                
    
    def _swap_axes(self, i: int, j: int):
        self.values = np.swapaxes(self.values, i, j)
        new_dims = list(self.dimensions)
        new_dims[i], new_dims[j] = new_dims[j], new_dims[i]
        self.dimensions = tuple(new_dims)
                
    
    def match_to_broadcast(self, other):
        """Moves and adds dimensions in `self` to be broadcastable with
        `other`. All the rightmost dimensions in `self` will be identical to
        the dimensions of `other`, and the dimensions unique to `self` will
        be the leftmost ones, which will be automatically broadcasted on
        `other` by numpy.
        
        Example input:
        ```
            self.dimensions  = (a, b, c)
            other.dimensions = (c, b, d)
        ```
        Result:
        ```
            self.dimensions  = (a, b, c, d)
        ```
        
        After this operation, any numpy operation such as `self + other`
        will broadcast `other` to add the dimensions `(a,)` to the left.
        Note that added dimensions in `self` (here, d) have a size of 1.
        """
        # Enumerate the dimensions in reverse order (i.e. starting from the right)
        for i_other, dim in enumerate(reversed(other.dimensions)):
            N = len(self.dimensions)
            try:
                i_self = self.dimensions.index(dim)
                assert i_self >= i_other
                # Swap axes in self
                self._swap_axes(i_self, N - i_other)
            except ValueError:
                # Add an axis in self
                self.dimensions = self.dimensions[:N - i_other] + (dim,) + self.dimensions[N - i_other:]
                self.values = np.expand_dims(self.values, N - i_other)
