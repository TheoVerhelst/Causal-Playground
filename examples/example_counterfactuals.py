from scipy import stats
from causality import Variable, Xor, IndependentDistribution, CausalModel
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

expression = EqualityExpr(Z.do(X, False), Y)

print("P(" + str(expression) + ") =", C.probability(expression))
