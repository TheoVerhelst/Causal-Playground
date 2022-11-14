import matplotlib.pyplot as plt
from random_system import generate_linear_system, sample_linear_system
from gaussian import GaussianIndependenceTest
from discovery import pc_algorithm
from plot import force_based_position, plot_graph

graph = generate_linear_system(
    n_nodes=3,
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
res = pc_algorithm(X, ind_test.gaussian_indep_test, alpha=0.05, initial_graph=graph.complete())
print(ind_test.corr_matrix)
force_based_position(res)

for n in graph.nodes():
    print("{} = {:.1f} += {:.1f}".format(n, graph.node(n)["mu"], graph.node(n)["sigma"]))
for e in graph.edges():
    print("{} -{:.1f}-> {}".format(e[0], e[2], e[1]))

plot_graph(res)
plt.show()

