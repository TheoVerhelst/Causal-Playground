import numpy as np

class Variable:
    def __init__(self, name: str, support: list, intervention = None):
        self.name = name
        self.support = support
        self.intervention = intervention
     
    def do(self, variable, value):
        return Variable(self.name, self.support, (variable, value))
    
    def __eq__(self, other):
        return other and self.__dict__ == other.__dict__
        
    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.name, self.support, self.intervention))
    
    def __str__(self):
        res = str(self.name)
        if self.intervention is not None:
            res += "_{" + str(self.intervention[0]) + " = " + str(self.intervention[1]) + "}"
        return res
    
    def __repr__(self):
        return str(self)
        
    def __lt__(self, other):
        """For topological sort."""
        return str(self) < str(other)
