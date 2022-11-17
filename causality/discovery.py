from itertools import combinations
from causality.causal_graph import CausalGraph

def pc_algorithm(data, indep_test, alpha, initial_graph=None, **kwargs):
    if initial_graph is None:
        graph = CausalGraph()
        for i in range(data.shape[1]):
            grap.add_node("V" + str(i))
        graph = graph.complete()
    else:
        graph = initial_graph

    nodes = graph.nodes()
    cond_set_size = 0
    sep_set = {}

    # Remove edges using conditional independence tests
    while max(graph.out_degree(x) for x in nodes) > cond_set_size:
        for x in nodes:
            adj_x = set(graph.nodes(from_node=x))
            for y in adj_x:
                adj_x_y = adj_x.difference({y})
                # combinations will return an empty list if cond_set_size > len(adj_x_y)
                for Z in combinations(adj_x_y, cond_set_size):
                    p = indep_test.indep_test(x, y, Z, **kwargs)
                    if p > alpha:
                        graph.del_edge(x, y)
                        graph.del_edge(y, x)
                        sep_set[(x, y)] = Z
                        sep_set[(y, x)] = Z
                        break
        cond_set_size += 1

    for (x, y) in combinations(nodes, 2):
        # Orient colliders
        if not graph.is_adjacent(x, y) and (x, y) in sep_set:
            rel_x = graph.undirected_neighbors({x})
            rel_y = graph.undirected_neighbors({y})
            for z in rel_x.intersection(rel_y):
                if z not in sep_set[(x, y)]:
                    # Orient as x -> z <- y
                    graph.del_edge(z, x)
                    graph.del_edge(z, y)

        # Orient undirected edges
        # Rules for remaining orientations, see e.g. Pearl 2000 sec. 2.5
        # R1: orient x - y as x -> y if there is z -> x with y not adjacent
        # to z
        if graph.is_undirected(x, y):
            parents_x = {z for z in graph.parents({x}) if not graph.is_undirected(x, z)}
            for z in parents_x:
                if not graph.is_adjacent(z, y):
                    graph.del_edge(y, x)
                    break

        # R2: orient x - y as x -> y if there is a chain x -> z -> y for
        # some z
        if graph.is_undirected(x, y):
            children_x = {z for z in graph.children({x}) if not graph.is_undirected(x, z)}
            parents_y = {z for z in graph.parents({y}) if not graph.is_undirected(y, z)}
            if not children_x.isdisjoint(parents_y):
                graph.del_edge(y, x)

        # R3: orient x - y as x -> y if there are two chains x - w -> y and
        # x - z -> y with w and z not adjacent
        if graph.is_undirected(x, y):
            for (z, w) in combinations(parents_y, 2):
                if graph.is_undirected(x, z) and graph.is_undirected(x, w):
                    graph.del_edge(y, x)
                    break

        # R4: orient x - y as x -> y if there are two chains x - w -> z and
        # w -> z -> y such that w and y are not adjacent and x and z are
        # adjacent
        if graph.is_undirected(x, y):
            for z in parents_y:
                if graph.is_adjacent(x, z):
                    for w in graph.undirected_neighbors(x):
                        if not graph.is_adjacent(w, y) and graph.is_directed(w, z):
                            graph.del_edge(y, x)
                            break
                    if graph.is_directed(x, y):
                        break
    return graph
    
