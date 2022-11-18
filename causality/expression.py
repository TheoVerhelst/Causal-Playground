from functools import reduce
from typing import Protocol, Sequence, Union
import numpy as np
from causality.variable import Variable
from causality.discrete_set import DiscreteSet

class Expression(Protocol):
    """Defines the expected methods (i.e.the protocol) of Expression
    objects.
    """
    def __str__(self) -> str:
        ...
    
    def values(self) -> DiscreteSet:
        """Returns a `DiscreteSet` indicating which valuations of the
        variables satisfy the expression. The set will contain one axis
        for each `Variable` involved in the expression.
        """
        ...

class EqualityExpr:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
    
    def __str__(self):
        return str(self.lhs) + " = " + str(self.rhs)
    
    def values(self):
        # If we have variable = literal or literal = variable
        if isinstance(self.lhs, Variable) != isinstance(self.rhs, Variable):
            if isinstance(self.lhs, Variable):
                variable = self.lhs
                value = self.rhs
            else:
                variable = self.rhs
                value = self.lhs
            values_array = np.full(len(variable.support), False)
            if value in variable.support:
                values_array[variable.support.index(value)] = True
            return DiscreteSet((variable,), values_array)
            
        elif isinstance(self.lhs, Variable):
            # We have variable1 = variable2
            values_array = np.full((len(self.lhs.support), len(self.rhs.support)), False)
            for i, val_lhs in enumerate(self.lhs.support):
                for j, val_rhs in enumerate(self.rhs.support):
                    if val_lhs == val_rhs:
                        values_array[i, j] = True
            return DiscreteSet((self.lhs, self.rhs), values_array)
            
        else:
            # We have value1 = value2. Return a zero-dimensional set
            return DiscreteSet((), self.lhs == self.rhs)
            

class ConjunctionExpr:
    def __init__(self, expressions):
        self.expressions = expressions
    
    def __str__(self):
        return ", ".join(str(e) for e in self.expressions)
    
    def values(self):
        return reduce(lambda a, b: a & b, [e.values() for e in self.expressions])
        
        
class DisjunctionExpr:
    def __init__(self, expressions):
        self.expressions = expressions
    
    def __str__(self):
        return " \\lor ".join(str(e) for e in self.expressions)
    
    def values(self):
        return reduce(lambda a, b: a | b, [e.values() for e in self.expressions])


class ExclusiveDisjunctionExpr:
    def __init__(self, expressions):
        self.expressions = expressions
    
    def __str__(self):
        return " \\oplus ".join(str(e) for e in self.expressions)
    
    def values(self):
        return reduce(lambda a, b: a ^ b, [e.values() for e in self.expressions])
        

class NegationExpr:
    def __init__(self, expression):
        self.expression = expression
    
    def __str__(self):
        if isinstance(self.expression, Equality):
            return str(self.expression.variable) + " \\neq " + str(self.expression.value)
        else:
            return "\\neg " + str(self.expression)
    
    def values(self):
        return ~self.expression.values()


class ProbabilityExpr:
    def __init__(self, expression, condition = None, intervention = None):
        self.expression = expression
        self.condition = condition
        self.intervention = intervention

    def __str__(self):
        res = "P(" + str(self.expression)
        
        if self.condition is not None or self.intervention is not None:
            res += " | "
        
        if self.condition is not None:
            res += str(self.condition)
        
        if self.intervention is not None:
            res += "do(" + str(self.intervention) + ")"
        
        res += ")"
        return res

    def __repr__(self):
        return str(self)


class SummationExpr:
    def __init__(self, indices, expression):
        self.indices = indices
        self.expression = expression

    def __str__(self):
        return "\\sum_{" + str(self.indices) + "} " + str(self.expression)

    def __repr__(self):
        return str(self)


class ProductExpr:
    def __init__(self, expressions):
        self.expressions = expressions

    def __str__(self):
        return " ".join(str(e) for e in self.expressions)

    def __repr__(self):
        return str(self)


def make_realisation(variables: Sequence[Union[Variable, str]]) -> set[str]:
    """Returns the set of variable names as string, in lowercase. This
    is used when summing over the values of variables, to define the
    summation indices.
    :param variables: A sequence of `Variable` or strings.
    """
    return set(V.name.lower() \
            if isinstance(V, Variable) \
            else V.lower() 
            for V in variables)


def make_prime(variables: Sequence[Union[Variable, str]]) -> set[Union[Variable, str]]:
    """Returns the set of variables where a prime is added to the name
    of the variables.
    :param variables: A sequence of `Variable` or strings.
    """
    return set(Variable(V.name + "'", V.support, V.intervention) \
            if isinstance(V, Variable) \
            else V + "'" \
            for V in variables)


def make_realisation_prime(variables: Sequence[Union[Variable, str]]) -> set[str]:
    """Returns the set of variable names in lowercase, with a prime.
    :param variables: A sequence of `Variable` or strings.
    """
    return set(V.name.lower() + "'" \
            if isinstance(V, Variable) \
            else V.lower() + "'" \
            for V in variables)

