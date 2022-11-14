from scipy import stats
from causality.variable import Variable
from causality.discrete_function import Xor
from causality.distribution import IndependentDistribution
from causality.causal_model import CausalModel
from causality.expression import ConjunctionExpr, EqualityExpr

X = Variable("X", (False, True))
Y = Variable("Y", (False, True))
Z = Variable("Z", (False, True))

P = IndependentDistribution({
    X: stats.bernoulli(0.2),
    Y: stats.bernoulli(0.4)
})

F = {Z: Xor((X, Y), Z)}

C = CausalModel(P, F)

expression = ConjunctionExpr([
    EqualityExpr(Z.do(X, False), True),
    EqualityExpr(Z.do(X, True), False)]
)
print("P(" + str(expression) + ") =", C.probability(expression))
