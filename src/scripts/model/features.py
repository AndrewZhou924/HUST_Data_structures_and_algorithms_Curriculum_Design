import networkx as nx
from scipy.spatial.distance import cosine

class CompareFeature(object):
    def __init__(self, graph1, graph2):
        self._value = (self._compute(graph1)==self._compute(graph2))

    def _compute(self, graph):
        # Should return a comparable for a graph such as int or list
        pass

    def get_value(self):
        # Returns true if both graphs have the same value
        return self._value


class CompareNumOfNodes(CompareFeature):
    def _compute(self, graph):
        return len(graph.nodes())


class CompareNumOfEdges(CompareFeature):
    def _compute(self, graph):
        return len(graph.edges())


class CompareDegreeDistribution(CompareFeature):
    def _compute(self, graph):
        degree = graph.degree().values()
        # degree.sort()
        sorted(degree)
        return degree


class CompareDirected(CompareFeature):
    def _compute(self, graph):
        return graph.is_directed()


class CompareLSpectrum(CompareFeature):
    def __init__(self, graph1, graph2):
        self._value = abs(cosine(self._compute(graph1), self._compute(graph2)))

    def _compute(self, graph):
        s = nx.laplacian_spectrum(graph)

        return s

class CompareASpectrum(CompareFeature):
    def __init__(self, graph1, graph2):
        self._value = abs(cosine(self._compute(graph1), self._compute(graph2)))

    def _compute(self, graph):
        s = nx.adjacency_spectrum(graph)
        return s

