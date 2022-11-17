from random import uniform, shuffle
from itertools import combinations
import numpy as np
from scipy.stats import norm
from causality.causal_graph import CausalGraph

def generate_linear_system(n_nodes, n_edges, min_mu, max_mu, min_sigma, max_sigma, min_rho, max_rho):
    if n_edges > (n_nodes - 1) * n_nodes / 2:
        raise ValueError("Can't create a DAG with {} nodes and {} edges".format(len(nodes), n_edges))

    nodes = ["V" + str(i) for i in range(n_nodes)]

    shuffle(nodes)
    # Create a list of all possible edges
    edges = list(combinations(nodes, 2))
    shuffle(edges)
    # Create a graph using only the n_edges first edges
    graph =  CausalGraph(from_list=edges[:n_edges])

    for node in nodes:
        graph.add_node(node, {
            "mu": uniform(min_mu, max_mu),
            "sigma": uniform(min_sigma, max_sigma)
        })

    for edge in graph.edges():
        begin, end, value = edge
        graph.add_edge(begin, end, value=uniform(min_rho, max_rho))

    return graph

def sample_linear_system(graph, n_samples):
    """Returns a matrix of shape (n_sample, len(graph.nodes())) containing
    values sampled according to the graph. The columns are ordered according
    to the node order given by graph.nodes().
    """
    unsorted_nodes = graph.nodes()
    sorted_nodes = list(graph.topological_sort())
    X = np.empty(shape=(n_samples, len(sorted_nodes)))
    for node in sorted_nodes:
        i = unsorted_nodes.index(node)
        X[:, i] = norm.rvs(
                loc = graph.node(node)["mu"],
                scale = graph.node(node)["sigma"],
                size = n_samples
        )
        for parent in graph.parents({node}):
            parent_idx = unsorted_nodes.index(parent)
            # This should always be true if the node list is topological
            assert sorted_nodes.index(parent) < sorted_nodes.index(node)
            # Add the value of the parent multiplied by the link coefficient
            X[:, i] += X[:, parent_idx] * graph.edge(parent, node)

    return X
