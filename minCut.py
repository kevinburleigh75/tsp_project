from copy import deepcopy
from collections import defaultdict
from itertools import chain
def find_connected_node(graph, node_set):
    """
    Finds the tightly connected node in V not in A 
    Input: graph with node 1 and node 2 keys and weight values,
    node_set: set representing A from min cut algorithm 
    Output: node that is tightly connected not in A 
    """
    max_weight = 0
    best_node = None
    all_nodes = graph.keys()
    for node1 in all_nodes:
        if node1 not in node_set:
            current_weight = 0
            for node2 in node_set:
                if node2 in graph[node1].keys():
                    current_weight += graph[node1][node2]
            if current_weight > max_weight: 
                best_node = node1
                max_weight = current_weight
    return best_node

def merge_nodes(old_graph, node_t, node_s):
    """
    merges string type nodes t and s and returns a modified copy of 
    the old graph with updated edge weights 
    """
    graph = deepcopy(old_graph)
    t_edges = graph[node_t]
    s_edges = graph[node_s]
    new_node = node_t + "--" + node_s
    #print t_edges
    #print s_edges

    new_node_edges = defaultdict(int)
    for k, v in chain(t_edges.items(), s_edges.items()):
        if k != node_s or k != node_t:
            new_node_edges[k] += v

    for node in graph.keys():
        if node not in [node_s, node_t]:
            if node_t in graph[node].keys() and node_s in graph[node].keys():
                graph[node][new_node] = graph[node][node_s] + graph[node][node_t]
                graph[node].pop(node_t)
                graph[node].pop(node_s)
            elif node_t in graph[node].keys():
                graph[node][new_node] = graph[node][node_t]
                graph[node].pop(node_t)
            elif node_s in graph[node].keys():
                graph[node][new_node] = graph[node][node_s]
                graph[node].pop(node_s)

    graph.pop(node_t)
    graph.pop(node_s)
    graph[new_node] = new_node_edges
    return graph 

def min_cut(graph):
    """
    Computes value of min cut and returns value and nodes in the min cut
    """
    min_cut = float("inf")
    min_cut_nodes = []
    while True: 
        node_set = [graph.keys()[0]]
        while len(node_set) < len(graph.keys()) - 1:
            node_set.append(find_connected_node(graph, node_set))
        current_cut = 0 
        penultimate_node = node_set[-1]
        last_node = find_connected_node(graph, node_set)
        current_cut_nodes = [last_node] 
        for node_end in graph[last_node].keys():
            current_cut += graph[last_node][node_end]
            current_cut_nodes.append(node_end)
        #print graph, penultimate_node, last_node
        graph = merge_nodes(graph, penultimate_node, last_node)
        if current_cut < min_cut:
            min_cut = current_cut
            min_cut_nodes = current_cut_nodes
        if len(graph.keys()) == 1: 
            break
    return min_cut, min_cut_nodes

if __name__ == '__main__':
    #connected graph with varying weights 
    #nodes are type string
    #distances are type int
    graph1 = {"1":{"2":1, "4":1,"5":2}, 
        "2":{"1":1, "3":1, "4":2,"5":1},
        "3":{"2":1, "4":2,"5":1},
        "4":{"2":2, "3":2, "1":1,"5":1},
        "5":{"2":1, "3":1, "4":1, "1":2}}
    graph2 = {"1":{"2":1, "4":1,"3":3}, 
        "2":{"1":1, "3":2},
        "3":{"1":3, "2":2},
        "4":{"1":1}} 


    #print find_connected_node(graph, set(["1","2","3","4"]))
    #print merge_nodes(graph1, "1","2")
    #print merge_nodes(graph2, "2", "1")
    #print min_cut(graph1)
    #print min_cut(graph2)
