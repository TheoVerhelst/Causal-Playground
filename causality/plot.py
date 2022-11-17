from math import sin, cos, atan2, sqrt
from numbers import Number
from random import uniform
from itertools import permutations
from causality.causal_graph import CausalGraph
from matplotlib.axes import Axes

def plot_graph(graph: CausalGraph, axis: Axes, font_size: int=13, font_family: str="serif"):
    plot_margin_x = 1
    plot_margin_y = 1
    gap_begin = 0.25
    gap_end = 0.25
    text_shift_x = -0.1
    text_shift_y = -0.1

    nodes_x = [graph.node(n)[0] for n in graph.nodes()]
    nodes_y = [graph.node(n)[1] for n in graph.nodes()]

    axis.set_xlim(min(nodes_x) - plot_margin_x, max(nodes_x) + plot_margin_x)
    axis.set_ylim(min(nodes_y) - plot_margin_y, max(nodes_y) + plot_margin_y)
    axis.axis('off')

    for node in graph.nodes():
        pos = graph.node(node)
        axis.text(
            pos[0] + text_shift_x,
            pos[1] + text_shift_y,
            node,
            fontfamily=font_family,
            fontsize=font_size
        )

    for begin, end, value in graph.edges():
        begin_pos = graph.node(begin)
        end_pos = graph.node(end)

        angle = atan2(end_pos[1] - begin_pos[1], end_pos[0] - begin_pos[0])
        arrow_x = begin_pos[0] + gap_begin * cos(angle)
        arrow_y = begin_pos[1] + gap_begin * sin(angle)
        dx = end_pos[0] - gap_end * cos(angle) - arrow_x
        dy = end_pos[1] - gap_end * sin(angle) - arrow_y

        axis.arrow(
            arrow_x, arrow_y,
            dx, dy,
            width=0.05,
            length_includes_head=True,
            color="black",
            linewidth=0
        )

def force_based_position(graph: CausalGraph, min_x: Number=0, max_x: Number=2, min_y: Number=0, max_y: Number=2):
    repulsion = 0.1
    attraction = 1
    spring_length = 2

    n = len(graph.nodes())
    pos = [[uniform(min_x, max_x), uniform(min_y, max_y)] for i in range(n)]
    vel = [[0, 0] for i in range(n)]

    max_iter = 100000
    epsilon = 0.01
    for _ in range(max_iter):
        # Compute forces
        forces = [[0, 0] for i in range(n)]
        for (i, j) in permutations(range(n), 2):
            dx = pos[i][0] - pos[j][0]
            dy = pos[i][1] - pos[j][1]
            dist2 = dx*dx + dy*dy
            dist2 = max(dist2, 0.01)
            F = repulsion / dist2**(3/2)
            forces[i][0] += F * dx
            forces[i][1] += F * dy
            if graph.edge(i, j) is not None:
                G = attraction * (1 - spring_length / sqrt(dist2))
                forces[i][0] -= G * dx
                forces[i][1] -= G * dy

        # Compute positions
        for i in range(n):
            for x in range(2):
                pos[i][x] += forces[i][x] * 0.01

        # Check convergence
        if max(max(f) for f in forces) <= epsilon:
            break

    for i, node in enumerate(graph.nodes()):
        graph.add_node(node, tuple(pos[i]))

    return graph
