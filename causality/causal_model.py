from functools import reduce
from copy import copy
from typing import Mapping
import numpy as np
from causality.causal_graph import CausalGraph
from causality.discrete_function import DiscreteFunction, ConstantFunction
from causality.variable import Variable
from causality.distribution import IndependentDistribution
from causality.expression import Expression

def infer_causal_graph(functions):
    G = CausalGraph()
    for Y, function in functions.items():
        for X in function.inputs:
            G.add_edge(X, Y)
    return G

class CausalModel:
    def __init__(
            self,
            exo_dist: IndependentDistribution,
            functions: Mapping[Variable, DiscreteFunction]):
        self.exo_dist = exo_dist
        self.functions = functions
        self.graph = infer_causal_graph(functions)
        self.twin_networks = set()
        self._update_sorted_endogenous()
    
    def _update_sorted_endogenous(self):
        # Check that we have a probability distribution on all nodes
        # without a parent, or that they are constant functions
        assert all(var in self.exo_dist.dists or isinstance(self.functions[var], ConstantFunction) \
                for var in self.graph.nodes(in_degree=0))
        self.sorted_endogenous = [v for v in self.graph.topological_sort() \
                if v in self.functions.keys()]
    
    def rvs(self, size: int) -> dict[Variable, np.ndarray]:
        # Step 1: all exogenous variables are sampled
        values = self.exo_dist.sample(size)
        
        # Step 2: all endogenous variables are iterated in topological order
        for var in self.sorted_endogenous:
            # Step 3: an endogenous variable is the output of a
            # deterministic function of its parents
            function = self.functions[var]
            assert all(parent in values for parent in function.inputs), \
                "The value of some parents of var is unknown"
            values[var] = function.function(*[values[parent] for parent in function.inputs])
        return values
    
    def probability(self, expression: Expression) -> float:
        values = expression.values()
        
        # If we have counterfactual variables, add the twin networks
        # to this model
        for var in values.dimensions:
            if var.intervention is not None and var.intervention not in self.twin_networks:
                self.add_twin_network(*var.intervention)
                
        # While there are endogenous variables in values
        while True:
            # Sort them in reversed topological order
            endogenous = [v for v in self.sorted_endogenous[::-1] \
                    if v in values.dimensions]
            
            if len(endogenous) == 0:
                break
                
            for var in endogenous:
                # The set of values that satisfy the functional
                # definition of var and the current set of values is
                # found with the tensor product where the sum is over
                # var. Trust me.
                values = values.tensor(self.functions[var].preimage, var)
                
        # Now we have the set of values of exogenous variables
        # that satisfy the expression, we only need to measure their probability
        return self.exo_dist.pmf(values)
    
    def intervention(self, var, value):
        self.twin_networks.add(var.intervention)
        res = copy(self)
        self.functions[var] = ConstantFunction(var, value)
        for pa in self.graph.parents({var}):
            self.graph.del_edge(pa, var)
    
    def add_twin_network(self, var, value):
        assert var in self.graph.nodes()
        V_x = {v: v.do(var, value) for v in self.graph.descendants({var})}
        for v, v_x in V_x.items():
            if v == var:
                self.functions[v_x] = ConstantFunction(v_x, value)
            else:
                F = self.functions[v]
                parents = tuple((V_x[pa] if pa in V_x else pa) for pa in F.inputs)
                for pa in parents:
                    self.graph.add_edge(pa, v_x)
                self.functions[v_x] = DiscreteFunction(F.function, parents, v_x)
        self._update_sorted_endogenous()
    
