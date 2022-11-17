from graph import Graph

class CausalGraph(Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._undirected = None
        self._complete = None
    
    def add_edge(self, *args, **kwargs):
        self._undirected = None
        self._complete = None
        super().add_edge(*args, **kwargs)
    
    def del_edge(self, *args, **kwargs):
        self._undirected = None
        self._complete = None
        super().del_edge(*args, **kwargs)
    
    def add_node(self, *args, **kwargs):
        self._undirected = None
        self._complete = None
        super().add_node(*args, **kwargs)
    
    def del_node(self, *args, **kwargs):
        self._undirected = None
        self._complete = None
        super().del_node(*args, **kwargs)
        
    def copy(self):
        return CausalGraph(from_dict=self.to_dict())

    def undirected(self):
        """Returns a copy of the graph with all edges duplicated in the opposite
        direction.
        """
        if self._undirected is None:
            self._undirected = self.copy()

            for begin, end, value in self.edges():
                if self.edge(end, begin) is None:
                    self._undirected.add_edge(end, begin, value)
                    
        return self._undirected

    def complete(self):
        """Returns a copy with every possible edges between nodes."""
        if self._complete is None:
            self._complete = CausalGraph()
            
            for node in self.nodes():
                self._complete.add_node(node, obj=self.node(node))

            for node_1 in self.nodes():
                for node_2 in self.nodes():
                    if node_1 != node_2:
                        self._complete.add_edge(node_1, node_2)

        return self._complete

    def all_undirected_paths(self, y, x):
        """Returns all paths between x and y disregarding edge direction."""
        return self.undirected().all_simple_paths(x, y)

    def is_collider(self, x, y, z):
        """True when x -> y <- z."""
        return self.edge(x, y) is not None and self.edge(z, y) is not None

    def is_chain(self, x, y, z):
        """True when x -> y -> z or x <- y <- z."""
        return (self.edge(x, y) is not None and self.edge(y, z) is not None) \
                or (self.edge(y, x) is not None and self.edge(z, y) is not None)

    def is_fork(self, x, y, z):
        """True when x <- y -> z."""
        return self.edge(y, x) is not None and self.edge(y, z) is not None

    def is_adjacent(self, x, y):
        """True if x -> y or x <- y."""
        return self.edge(x, y) is not None or self.edge(y, x) is not None

    def is_undirected(self, x, y):
        """True if x -> y and x <- y."""
        return self.edge(x, y) is not None and self.edge(y, x) is not None

    def is_directed(self, x, y):
        """True if x -> y and not x <- y."""
        return self.edge(x, y) is not None and self.edge(y, x) is not None

    def undirected_neighbors(self, X):
        """List all nodes adjacent to X but only with undirected edges."""
        return set().union(*({y for y in self.neighbors({x}) if self.is_undirected(x, y)} for x in X))

    def children(self, X):
        """Direct children of all members of X."""
        # the from_node parameter in graph.edges does not seem to work here
        return set().union(*(set(edge[1] for edge in self.edges() if edge[0] == x) for x in X))

    def parents(self, X):
        """Direct parents of all members of X."""
        # the to_node parameter in graph.edges does not seem to work here
        return set().union(*(set(edge[0] for edge in self.edges() if edge[1] == x) for x in X))

    def _neighbors_recursive(self, X, relation):
        res = set()
        new_members = X
        while len(new_members) > 0:
            res.update(new_members)
            new_members = relation(new_members).difference(res)
        return res

    def descendants(self, X):
        """All descendants of all members of X, with X included."""
        return self._neighbors_recursive(X, self.children)

    def ancestors(self, X):
        """All ancestors of all members of X, with X included."""
        return self._neighbors_recursive(X, self.parents)

    def neighbors(self, X):
        """All parents or children of all members of X."""
        return self.parents(X).union(self.children(X))

    def is_d_separated(self, X, Y, Z):
        """True if X is d-separated from Y given Z."""
        Z = set(Z)

        if self.has_cycles():
            raise RuntimeError("Requires an acyclic graph")

        # X and Y are d-separated by Z if all paths are blocked.
        # If we find a non-blocked path, we can stop and return False.
        for x in X:
            for y in Y:
                for path in self.all_undirected_paths(x, y):
                    path_blocked = False
                    # Iterate over all nodes in the path
                    for i in range(1, len(path) - 1):
                        a, b, c = path[i - 1:i + 2]
                        if self.is_collider(a, b, c):
                            # If the middle node or its descendants do not contain
                            # any node from Z, then the collider blocks the path
                            if self.descendants({b}).isdisjoint(Z):
                                path_blocked = True
                                break
                        # We have a chain or a fork, check if b is in Z
                        elif b in Z:
                            path_blocked = True
                            break
                    if not path_blocked:
                        return False
        return True

    def remove_out_of(self, X):
        """Returns a copy of the graph with all edges going out of X removed."""
        return self._graph_surgery(X, into_X=False)

    def remove_into(self, X):
        """Returns a copy of the graph with all edges going into X removed."""
        return self._graph_surgery(X, into_X=True)

    def _graph_surgery(self, X, into_X):
        res = self.copy()
        for x in X:
            kwargs = {"to_node": x} if into_X else {"from_node": x}
            for edge in self.edges(**kwargs):
                res.del_edge(edge[0], edge[1])
        return res
