import matplotlib.pyplot as plt
from plot import plot_graph
from causality_graph import CausalityGraph
from identification import closed_form
from expression import Probability

g = CausalityGraph()
g.add_node("X", (0, 3))
g.add_node("Y", (0, 0))
g.add_node("Z1", (0, 1.5))
g.add_node("Z2", (1, 1.5))
g.add_node("U1", (-1, 1.5))
g.add_node("U2", (1, 3))
g.add_node("U3", (1, 0))

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

plt.rcParams['text.usetex'] = True
font_size = 13

fig, ax = plt.subplots()
fig.set_tight_layout(True)

plot_graph(g, ax, font_size)

x_base = 1.5
y_base = 3

text = ax.text(
    x_base, y_base,
    "$$" + str(Probability({"Y"}, inter={"X"})) + "$$",
    fontsize=font_size
)

bbox = text.get_tightbbox(fig.canvas.get_renderer())
text_topright = ax.transData.inverted().transform(bbox.max)
text_bottomleft = ax.transData.inverted().transform(bbox.min)

y_offset = (text_topright[1] - text_bottomleft[1]) * 3

for i, formula in enumerate(closed_forms):
    plt.text(
        text_topright[0] + 0.4,
        text_topright[1] - y_offset * i - 0.2,
        "$$=" + str(formula) + "$$",
        fontsize=font_size
    )

if len(closed_forms) == 0:
    plt.text(
        text_topright[0],
        text_topright[1],
        "$$\\textrm{is not identifiable}$$",
        fontsize=font_size
    )

fig.set_figwidth(9)
fig.set_figheight(5)
plt.show()

