from itertools import combinations
from causality.expression import ProbabilityExpr, SummationExpr, ProductExpr, make_prime

def no_back_door_path(graph, X, Y, Z):
    """True if there is no back-door path (confounding) from X to Y given Z."""
    return graph.remove_out_of(X).is_d_separated(X, Y, Z)

# "Causality" by Pearl (2009), def 3.3.1, p. 79.
def back_door_criterion(graph, X, Y, Z):
    return no_back_door_path(graph, X, Y, Z) and graph.descendants(X).isdisjoint(Z)

# Van der Zander, Benito, and Maciej Liśkiewicz. "Finding minimal d-separators
# in linear time and applications." Uncertainty in Artificial Intelligence.
# PMLR, 2020.
# There is probably a better implementation based on the Dijsktra algorithm,
# but I'm too lazy to write it
def closure(graph, X, A, Z):
    """
    Returns all nodes V for which there is a path from X to V that
        * contains only members of A, and
        * has no fork or chain in Z.
    """
    res = X.copy()
    for x in X:
        for a in A.difference(X):
            for path in graph.all_undirected_paths(x, a):
                valid_path = True
                for i, v in enumerate(path[1:-1]):
                    # If we are outside of A, this path does not work, go to the next one
                    if v not in A:
                        valid_path = False
                    if v in Z and not graph.is_collider(path[i-1], v, path[i + 1]):
                        valid_path = False
                if valid_path:
                    res.add(a)
                    break
    return res


# van der Zander, Benito, and Maciej Liśkiewicz. "Finding minimal d-separators
# in linear time and applications." Uncertainty in Artificial Intelligence.
# PMLR, 2020.
def blocking_set(graph, X, Y, U = set(), always_included = set()):
    R = set(graph.nodes()).difference(U)
    A = graph.ancestors(X.union(Y).union(always_included))
    Z_0 = R.intersection(A.difference(X.union(Y)))
    X_star = closure(graph, X, A, Z_0)
    Z_X = Z_0.intersection(X_star.union(always_included))
    Y_star = closure(graph, Y, A, Z_X)
    if not X_star.isdisjoint(Y):
        return None
    else:
        return Z_X.intersection(Y_star.union(always_included))


# This is exp time, see "Adjustment Criteria in Causal Diagrams: An Algorithmic
# Perspective" by Johannes Textor and Maciej Liskiewicz for a better
# algorithm (polynomial per output element).
def all_minimal_adjustment_sets(graph, X, Y, U = set()):
    candidates = set(graph.nodes()).difference(X).difference(Y).difference(U)
    res = set()
    for k in range(len(candidates)):
        for adjustment in combinations(candidates, k):
            adjustment = frozenset(adjustment)
            if back_door_criterion(graph, X, Y, adjustment):
                res.add(adjustment)
        if len(res) > 0:
            break
    return res

# "Causality" by Pearl (2009), sec. 4.3.3, p.117.
def closed_form(graph, X, Y, U = set()):
    """Find closed-form expressions for the causal effect P(Y | do(X)),
    considering U as latent (can't be adjusted for or conditioned on).

    Not every possible expression are returned, since there is an exponential
    number of them (for example, there can be many valid adjustment sets).
    """
    res = []

    # If there is no causal path from X to Y
    if graph.remove_into(X).is_d_separated(X, Y, set()):
        res.append(ProbabilityExpr(Y))
        return res

    # If there is no confounding from X to Y
    if no_back_door_path(graph, X, Y, set()):
        res.append(ProbabilityExpr(Y, cond=X))
        return res

    # Back-door adjustment
    for B in all_minimal_adjustment_sets(graph, X, Y, U=U):
        if len(B) > 0:
            closed_forms_B = closed_form(graph, X, B)
            for closed_form_B in closed_forms_B:
                res.append(
                    SummationExpr(
                        B,
                        ProductExpr([ProbabilityExpr(Y, cond=B.union(X)), closed_form_B])
                    )
                )

    # Front door adjustment
    Z_1 = graph.children(X).intersection(graph.ancestors(Y))
    if len(Z_1) > 0 and Y.isdisjoint(Z_1):
        # If Z_1 is unconfounded
        if no_back_door_path(graph.remove_into(X), Z_1, Y, set()) \
                and no_back_door_path(graph, X, Z_1, set()):
            # Front-door adjustment with unconfounded mediator Z_1
            res.append(Summation(
                Z_1,
                ProductExpr([
                    ProbabilityExpr(Z_1, cond=X),
                    SummationExpr(
                        make_prime(X),
                        ProductExpr([
                            ProbabilityExpr(Y, cond=make_prime(X).union(Z_1)),
                            ProbabilityExpr(make_prime(X))
                        ])
                    )
                ])
            ))

        else:
            # Generalized front-door where we adjust for confounding with the mediator Z_1
            Z_2_already_used = set()
            for Z_3 in all_minimal_adjustment_sets(graph, X, Z_1, U=U):
                for Z_4 in all_minimal_adjustment_sets(graph.remove_into(X), Z_1, Y, U=U):
                    Z_2 = Z_3.union(Z_4)
                    if X.isdisjoint(Z_2) and Z_2 not in Z_2_already_used:
                        Z_2_already_used.add(Z_2)
                        res.append(
                            SummationExpr(
                                Z_1.union(Z_2),
                                ProductExpr([
                                    ProbabilityExpr(Z_2),
                                    ProbabilityExpr(Z_1, cond=X.union(Z_2)),
                                    SummationExpr(
                                        make_prime(X),
                                        ProductExpr([
                                            ProbabilityExpr(Y, cond=make_prime(X).union(Z_1).union(Z_2)),
                                            ProbabilityExpr(make_prime(X), cond=Z_2)
                                        ])
                                    )
                                ])
                            )
                        )
    return res
