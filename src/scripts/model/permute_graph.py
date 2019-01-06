import random

def permute_graph(graph, permutations=1):
    # Perform pair-wise permutations on the graph to generate an isomorphic graph
    for i in range(permutations):
        [node1, node2] = random.sample(graph.nodes(), 2)

        # exchange node1 and node2
        temp = graph.edge[node1]
        graph.edge[node1] = graph.edge[node2]
        graph.edge[node2] = temp
    return graph


