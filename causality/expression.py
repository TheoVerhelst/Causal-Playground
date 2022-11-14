from functools import reduce
from typing import Protocol
from causality.discrete_set import DiscreteSet

class Expression(Protocol):
    def __str__(self) -> str:
        ...
    
    def values(self) -> DiscreteSet:
        ...

class EqualityExpr:
    def __init__(self, variable, value):
        self.variable = variable
        self.value = value
    
    def __str__(self):
        return str(self.variable) + " = " + str(self.value)
    
    def values(self):
        values = self.variable.values()
        if self.value in self.variable.support:
            values.values[self.variable.support.index(self.value)] = True
        return values


class InterventionExpr:
    def __init__(self, expression):
        self.expression = expression
    
    def __str__(self):
        return "\\mathrm{do}(" + str(self.expression) + ")"


class ConjunctionExpr:
    def __init__(self, expressions):
        self.expressions = expressions
    
    def __str__(self):
        return ", ".join(str(e) for e in self.expressions)
    
    def values(self):
        return reduce(lambda a, b: a.values() & b.values(), self.expressions)
        
        
class DisjunctionExpr:
    def __init__(self, expressions):
        self.expressions = expressions
    
    def __str__(self):
        return " \\lor ".join(str(e) for e in self.expressions)
    
    def values(self):
        return reduce(lambda a, b: a.values() | b.values(), self.expressions)


class ExclusiveDisjunctionExpr:
    def __init__(self, expressions):
        self.expressions = expressions
    
    def __str__(self):
        return " \\oplus ".join(str(e) for e in self.expressions)
    
    def values(self):
        return reduce(lambda a, b: a.values() ^ b.values(), self.expressions)
        

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
            res += " \\mid "
        
        if self.condition is not None:
            res += str(self.condition)
        
        if self.intervention is not None:
            res += str(self.intervention)
        
        res += ")"
        return res

    def __repr__(self):
        return str(self)


class SummationExpr:
    def __init__(self, indices, expression):
        self.indices = indices
        self.expression = expression

    def __str__(self):
        return "\\sum_{" + std(self.indices) + "} " + str(self.expression)

    def __repr__(self):
        return str(self)


class ProductExpr:
    def __init__(self, expressions):
        self.expressions = expressions

    def __str__(self):
        return " ".join(str(e) for e in self.expressions)

    def __repr__(self):
        return str(self)


def make_realisation(variables):
    return [V.lower() for V in variables]


def make_prime(variables):
    return [V + "'" for V in variables]


def make_realisation_prime(variables):
    return [V.lower() + "'" for V in variables]

