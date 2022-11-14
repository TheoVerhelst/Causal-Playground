from functools import reduce
import numpy as np
from causality.discrete_set import DiscreteSet

class DiscreteFunction:
    def __init__(self, function, inputs, output):
        """Constructor.
        
        :param function: function that takes as many arguments as the
            number of variables in `inputs`
        :param inputs: a sequence of `Variable` instances that indicate
            the function domain
        :param output: a `Variable` instance representing the function
            codomain
        """
        self.function = function
        self.inputs = tuple(inputs)
        self.output = output
        self.total_variables = self.inputs + (output,)
        self.total_dim = tuple(len(v.support) for v in self.total_variables)
        self.preimage = self._compute_preimage()
        
    def _compute_preimage(self):
        res = DiscreteSet(self.total_variables, np.full(self.total_dim, False))
        with np.nditer(res.values, flags=["multi_index"], op_flags=["readwrite"]) as it:
            for value in it:
                values = [res.dimensions[dim].support[val_i] \
                    for dim, val_i in enumerate(it.multi_index)]
                input_values = values[:-1]
                output_value = values[-1]
                image = self.function(*input_values)
                if output_value == image:
                    res.values[it.multi_index] = True
        return res


class Xor(DiscreteFunction):
    def __init__(self, inputs, output):
        super(Xor, self).__init__(
            lambda *values: reduce(lambda a, b: np.logical_xor(a, b), values),
            inputs,
            output
        )

class ConstantFunction(DiscreteFunction):
    def __init__(self, output, value):
        super(ConstantFunction, self).__init__(
            lambda: value,
            (),
            output
        )
