from collections import defaultdict
import random
from collections import deque

def solution_to_edges(binaryArray, numNodes):
    """
    converts binary array representing
    edges to an array of enumerated edges
    Input: binaryArray -- binary arrray with indices corresponding to edges,
           numNodes -- number of nodes in graph
    Output: array of edges in order

    """
    edges = []
    allEdges = []
    for i in range(1,numNodes):
        for j in range(i, numNodes+1):
            if i != j:
                allEdges.append((i,j))

    for i in range(0, len(binaryArray)):
        if binaryArray[i] == 1:
            edges.append(allEdges[i])
    return edges

def count_edge_nodes(edges, numNodes):
    """
    computes number of times a node apears in a list of tuple edges
    Input: edges -- array of integer tuples,
           numNodes -- number of nodes
    Output: dictionary with nodes and number of times they appear
    """
    dictionary = defaultdict(int)
    for num1, num2 in edges:
        dictionary[num1] += 1
        dictionary[num2] += 1
    return dictionary

def bfs(g, startnode):
    """
    Perform a breadth-first search on g starting at node startnode.

    Arguments:
    g -- undirected graph
    startnode - node in g to start the search from

    Returns:
    The distances from startnode to each node.
    """
    dist = {}

    # Initialize distances and predecessors
    for node in g:
        dist[node] = float('inf')
    dist[startnode] = 0

    # Initialize the search queue
    queue = deque([startnode])

    # Loop until all connected nodes have been explored
    while queue:
        j = queue.popleft()
        for h in g[j]:
            if dist[h] == float('inf'):
                dist[h] = dist[j] + 1
                queue.append(h)
    return dist

def connected_components(g):
    """
    Find all connected components in g.

    Arguments:
    g -- undirected graph

    Returns:
    A list of sets where each set is all the nodes in
    a connected component.
    """
    # Initially we have no components and all nodes remain to be
    # explored.
    components = []
    remaining = set(g.keys())

    while remaining:
        # Randomly select a remaining node and find all nodes
        # connected to that node
        node = random.choice(list(remaining))
        distances = bfs(g, node)
        visited = set()
        for i in remaining:
            if distances[i] != float('inf'):
                visited.add(i)
        components.append(visited)

        # Remove all nodes in this component from the remaining
        # nodes
        remaining -= visited
    return components

def make_Graph_from_Edges(edges):
    """
    makes a dictionary Graph from an array of tuple edges
    Input: edges -- array of tuples
    Output: dictionary of sets
    """
    graph = defaultdict(set)
    for edges in edges:

        node1 = edges[0]
        node2 = edges[1]
        graph[node1].add(node2)
        graph[node2].add(node1)

    return graph

def is_tour(solution, graph):
    """
    determines whether a given solution is a binary
    represenation of edges in a tour
    Input:  solution -- array of ints
            graph    -- dictionary[node_id] = connected_node_ids
    Output: True   if solution forms a tour
            False  otherwise
    """

    ##
    ## Check if solution is binary.
    ##

    for edge in solution:
        if edge not in [0.0,1.0]:
            print "Entries in our solution must be binary floats"
            return False

    ##
    ## Check if the number of selected edges is correct.
    ##

    num_nodes = len(graph.keys())
    edges = solution_to_edges(solution, num_nodes)
    if len(edges) != num_nodes:
        print "Number of edges must be the number of nodes for a tour"
        return False

    degree_by_node = count_edge_nodes(edges, num_nodes)
    for node in degree_by_node.keys():
        if degree_by_node[node] != 2:
            print "a node's degree must be 2 in a tour"
            return False

    pathGraph = make_Graph_from_Edges(edges)
    #print pathGraph
    pathComponents = connected_components(pathGraph)
    if len(pathComponents) > 1:
        print "Tour has more than one connected component"
        return False

    return True


if __name__ == '__main__':
    def makeGraph(num_nodes):
        """
        makes undirected connected graph
        Input:  num_nodes -- number of desired nodes
        Output: graph with num_nodes nodes
                A graph is a dictionary that takes a
                node id as a key and returns a set of
                connected node ids.  No loops allowed.
        """
        graph = defaultdict(set)
        for nid1 in range(1, num_nodes+1):
            for nid2 in range(1, num_nodes+1):
                if nid1 != nid2:
                    graph[nid1].add(nid2)
        return graph

    ##
    ## Selected edges form two connected components.
    ##

    gg = makeGraph(6)
    xx = [1.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0]
    assert is_tour(xx,gg) == False

    ##
    ## Selected edges form a tour.
    ##

    gg = makeGraph(3)
    xx = [1.0,1.0,1.0]
    assert is_tour(xx,gg) == True

    ##
    ## Not enough edges selected.
    ##

    gg = makeGraph(3)
    xx = [1.0,0.0,1.0]
    assert is_tour(xx,gg) == False

    ##
    ## Non-binary selections.
    ##

    gg = makeGraph(4)
    xx = [0.5,0.5,1.0,0.5,1.0,1.0]
    assert is_tour(xx,gg) == False




