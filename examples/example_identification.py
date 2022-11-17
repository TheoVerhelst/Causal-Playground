import matplotlib.pyplot as plt
from causality import CausalGraph
from causality.plot import plot_graph
from causality.identification import closed_form
from causality.expression import ProbabilityExpr

g = CausalGraph()
g.add_node("X",  (0,  3))
g.add_node("Y",  (0,  0))
g.add_node("Z1", (0,  1.5))
g.add_node("Z2", (1,  1.5))
g.add_node("U1", (-1, 1.5))
g.add_node("U2", (1,  3))
g.add_node("U3", (1,  0))

g.add_edge("X", "Z1")
g.add_edge("Z1", "Y")
g.add_edge("Z2", "Z1")
g.add_edge("U1", "X")
g.add_edge("U1", "Y")
g.add_edge("U2", "X")
g.add_edge("U2", "Z2")
g.add_edge("U3", "Z2")
g.add_edge("U3", "Y")

closed_forms = closed_form(g, {"X"}, {"Y"}, U={"U1", "U2", "U3"})

for formula in closed_forms:
    print(ProbabilityExpr("Y", intervention="X"), "=", formula)

if len(closed_forms) == 0:
    print(ProbabilityExpr("Y", intervention="X"), "is not identifiable")

fig, ax = plt.subplots()
plot_graph(g, ax)
plt.show()

