from itertools import permutations

import networkx as nx


class GraphBenchmark:
    params = ["Graph", "DiGraph", "MultiGraph", "MultiDiGraph"]
    param_names = ["graph_type"]

    def setup(self, graph_type):
        self.nodes = list(range(1, 1000))
        self.edges = [(u, v) for u in range(100) for v in range(80, 120)]
        self.weighted_edges = [(u, v, 1.1) for u in range(100) for v in range(80, 120)]
        self.subgraph_nodes = list(range(1, 100))
        self.subgraph_nodes_large = list(range(1, 900))
        self.null_G = getattr(nx, graph_type)()

        self.G = self.null_G.copy()
        self.G.add_nodes_from(self.nodes)
        self.G.add_edges_from(self.edges)

        self.weighted_G = self.null_G.copy()
        self.weighted_G.add_nodes_from(self.nodes)
        self.weighted_G.add_weighted_edges_from(self.weighted_edges)

    def time_graph_create(self, graph_type):
        _ = getattr(nx, graph_type)()

    def time_add_nodes_from(self, graph_type):
        self.G.add_nodes_from(self.nodes)

    def time_add_edges_from(self, graph_type):
        self.null_G.add_edges_from(self.edges)

    def time_add_weightededges_from(self, graph_type):
        self.null_G.add_weighted_edges_from(self.weighted_edges)

    def time_remove_nodes_from(self, graph_type):
        self.G.remove_nodes_from(self.nodes)

    def time_remove_edges_from(self, graph_type):
        self.G.remove_edges_from(self.edges)

    def time_copy(self, graph_type):
        _ = self.G.copy()

    def time_to_directed(self, graph_type):
        _ = self.G.to_directed()

    def time_to_undirected(self, graph_type):
        _ = self.G.to_undirected()

    def time_subgraph(self, graph_type):
        _ = self.G.subgraph(self.subgraph_nodes).copy()

    def time_subgraph_large(self, graph_type):
        _ = self.G.subgraph(self.subgraph_nodes_large).copy()

    def time_number_of_edges(self, graph_type):
        _ = self.G.number_of_edges()

    def time_size(self, graph_type):
        _ = self.G.size()

    def time_weighted_size(self, graph_type):
        _ = self.G.size(weight="weight")
