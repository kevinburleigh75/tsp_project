#!/usr/bin/env python

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

def create_random_nodes(NN=10):
    nodes = np.arange(0,NN)
    positions = {}
    for node in nodes:
        positions[node] = np.random.rand(2)

    return (nodes, positions)


def create_random_edges(nodes, NE_max=100):
    NN     = len(nodes)
    NE_max = np.min([NN*(NN-1)/2, NE_max])

    edges = []
    for _ in xrange(NE_max):
        edge_nodes = sorted(random.sample(nodes, 2))
        edges.append(tuple(edge_nodes))

    edges = set(edges)
    return edges


def main():
    nodes, positions = create_random_nodes(NN=10)
    edges            = create_random_edges(nodes, 3)

    graph = nx.Graph()
    graph.add_nodes_from(nodes)

    for _ in xrange(100):
        graph.remove_edges_from(graph.edges())
        edges = create_random_edges(nodes, 3)
        graph.add_edges_from(edges)
        plt.cla()
        axes = plt.gca()
        axes.set_xlim([0,1])
        axes.set_ylim([0,1])
        nx.draw(graph, pos=positions, hold=True)
        plt.pause(0.05)




if __name__ == '__main__':
    main()


