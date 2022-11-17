import matplotlib.pyplot as plt
from causality.random_system import generate_linear_system, sample_linear_system
from causality.gaussian import GaussianIndependenceTest
from causality.discovery import pc_algorithm
from causality.plot import force_based_position, plot_graph

graph = generate_linear_system(
    n_nodes=5,
    n_edges=3,
    min_mu=0,
    max_mu=0,
    min_sigma=0.1,
    max_sigma=0.1,
    min_rho=1,
    max_rho=1
)
N = 100000
X = sample_linear_system(graph, N)
ind_test = GaussianIndependenceTest(X, graph.nodes())
res = pc_algorithm(X, ind_test, alpha=0.05, initial_graph=graph.complete())
force_based_position(res)

fig, ax = plt.subplots()
plot_graph(res, ax)
plt.show()

