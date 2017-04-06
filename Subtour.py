from collections import defaultdict

def binary_to_edge(binaryArray, numNodes):
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

def isSubtour(solution, graph):
	"""
	determines whether a given solution is a binary 
	represenation of edges in a cycle, and whether those 
	edges are part of a cycle
	Input: solution -- array of ints , graph -- dictionary of sets 
	Output: True if solution forms a cycle, False otherwise

	"""
	node_Degree = 2
	numNodes = len(graph.keys())
	for edge in solution:
		if edge not in [0,1]:
			print "Entries in our solution must be binary"
			return False
	edges = binary_to_edge(solution, numNodes)

	if len(edges) != numNodes:
		print "Number of edges must be the number of nodes for a cycle"
		return False
	nodeInstanceDictionary = count_edge_nodes(edges, numNodes)
	for node in nodeInstanceDictionary.keys():
		if nodeInstanceDictionary[node] != node_Degree:
			print "a node's degree must be 2 in a cycle"
			return False 
	return True

def makeGraph(numNodes):
	"""makes undirected connected graph
	Input: numNodes -- number of desired nodes
	Output: graph with numNodes nodes 
	"""
	graph = defaultdict(set)
	for i in range(1, numNodes+1):
		for j in range(1, numNodes+1):
			if i != j: 
				graph[i].add(j)
	return graph





