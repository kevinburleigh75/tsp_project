from collections import deque
import copy

class Graph(object):

    def __init__(self, nodes, edges):
        self.nodes        = nodes
        self.edges        = edges

        self.node_by_idx = {idx: node for idx,node in enumerate(self.nodes)}
        self.idx_by_node = {node: idx for idx,node in enumerate(self.nodes)}

        self.edge_by_idx = {idx: edge for idx,edge in enumerate(self.edges)}
        self.idx_by_edge = {edge: idx for idx,edge in enumerate(self.edges)}

        self.edges_by_node = {node: set() for node in self.nodes}
        self.nodes_by_node = {node: set() for node in self.nodes}
        for edge in self.edges:
            (node1,node2) = edge

            self.edges_by_node[node1].add(edge)
            self.edges_by_node[node2].add(edge)

            self.nodes_by_node[node1].add(node2)
            self.nodes_by_node[node2].add(node1)


    def connected_components(self):
        connected_components_nodes = []
        connected_components_edges = []

        unvisited_nodes = set(self.nodes)
        while len(unvisited_nodes) != 0:
            queue = deque([unvisited_nodes.pop()])

            visited_nodes = set()
            visited_edges = set()
            while len(queue) != 0:
                cur_node = queue.popleft()

                if cur_node in unvisited_nodes:
                    unvisited_nodes.remove(cur_node)

                visited_nodes.add(cur_node)

                for edge in self.edges_by_node[cur_node]:
                    visited_edges.add(edge)

                    (node1,node2) = edge
                    if node1 not in visited_nodes:
                        queue.append(node1)
                    if node2 not in visited_nodes:
                        queue.append(node2)

            connected_components_nodes.append(visited_nodes)
            connected_components_edges.append(visited_edges)

        return connected_components_nodes, connected_components_edges


    def binary_connected_components(self, solution):
        binary_one_edges = [edge for edge in self.edges if solution[self.idx_by_edge[edge]] == 1.0]
        binary_one_graph = Graph(nodes=self.nodes, edges=binary_one_edges)

        ccs_nodes, ccs_edges = binary_one_graph.connected_components()
        return (ccs_nodes, ccs_edges)


        # selected_edges_by_node = {node: set() for node in self.nodes}
        # for idx,value in enumerate(solution):
        #     if value == 1.0:
        #         edge = self.edge_by_idx[idx]
        #         (node1,node2) = edge
        #         selected_edges_by_node[node1].add(edge)
        #         selected_edges_by_node[node2].add(edge)

        # connected_component_nodes = []
        # connected_component_edges = []

        # unvisited_nodes = set(self.nodes)
        # while len(unvisited_nodes) != 0:
        #     queue = deque([unvisited_nodes.pop()])

        #     visited_nodes = set()
        #     visited_edges = set()
        #     while len(queue) != 0:
        #         cur_node = queue.popleft()

        #         if cur_node in unvisited_nodes:
        #             unvisited_nodes.remove(cur_node)

        #         visited_nodes.add(cur_node)

        #         for edge in selected_edges_by_node[cur_node]:
        #             visited_edges.add(edge)

        #             (node1,node2) = edge
        #             if node1 not in visited_nodes:
        #                 queue.append(node1)
        #             if node2 not in visited_nodes:
        #                 queue.append(node2)

        #     connected_component_nodes.append(visited_nodes)
        #     connected_component_edges.append(visited_edges)

        # return connected_component_nodes, connected_component_edges


    def get_cut_edges(self, nodes):
        nodes = set(nodes)
        edges = set()
        for node in nodes:
            for edge in self.edges_by_node[node]:
                if (edge[0] not in nodes) or (edge[1] not in nodes):
                    edges.add(edge)
        return edges


    def is_tour(self, solution):
        ##
        ## Check that the solution is a "binary" vector,
        ## and that number of selected edges is the same as
        ## the number of nodes
        ##

        num_selected_edges = 0
        for idx,value in enumerate(solution):
            if not value in [0.0,1.0]:
                # print('non-binary')
                return False

            if value == 1.0:
                num_selected_edges += 1

        if num_selected_edges != len(self.nodes):
            # print('incorrect number of selected edges')
            return False

        ##
        ## Form a mapping from a node to nodes connected
        ## by selected edges, and ensure that each node
        ## has degree 2.
        ##

        selected_edges_by_node  = {node: set() for node in self.nodes}
        for idx,value in enumerate(solution):
            if value == 1.0:
                edge = self.edge_by_idx[idx]
                (node1,node2) = edge
                selected_edges_by_node[node1].add(edge)
                selected_edges_by_node[node2].add(edge)

        for node,selected_edges in selected_edges_by_node.items():
            if len(selected_edges) != 2:
                # print('degree violation')
                return False

        ##
        ## Find all connected components.  If there is only one,
        ## we have a tour.
        ##

        node_components, edge_components = self.binary_connected_components(solution)
        if len(node_components) != 1:
            # print('number of components = {}'.format(len(node_components)))
            # print(node_components)
            # print('subtours')
            return False

        # print('valid tour')
        return True

if __name__ == '__main__':

    nodes = ['a', 'b', 'c', 'd', 'e', 'f']
    edges = [
        ('a', 'b'),
        ('a', 'f'),
        ('b', 'c'),
        ('b', 'd'),
        ('b', 'e'),
        ('b', 'f'),
        ('c', 'd'),
        ('c', 'e'),
        ('c', 'f'),
        ('d', 'e'),
        ('d', 'f'),
        ('e', 'f'),
    ]

    graph = Graph(nodes=nodes, edges=edges)

    ##
    ## non-binary solution with correct "degree"
    ##

    xx = [
        1.0, ## ('a', 'b')
        1.0, ## ('a', 'f')
        0.5, ## ('b', 'c')
        0.0, ## ('b', 'd')
        0.5, ## ('b', 'e')
        0.0, ## ('b', 'f')
        1.0, ## ('c', 'd')
        0.0, ## ('c', 'e')
        0.5, ## ('c', 'f')
        1.0, ## ('d', 'e')
        0.0, ## ('d', 'f')
        0.5, ## ('e', 'f')
    ]

    assert graph.is_tour(xx) == False

    ##
    ## subtours
    ##

    xx = [
        1.0, ## ('a', 'b')
        1.0, ## ('a', 'f')
        0.0, ## ('b', 'c')
        0.0, ## ('b', 'd')
        0.0, ## ('b', 'e')
        1.0, ## ('b', 'f')
        1.0, ## ('c', 'd')
        1.0, ## ('c', 'e')
        0.0, ## ('c', 'f')
        1.0, ## ('d', 'e')
        0.0, ## ('d', 'f')
        0.0, ## ('e', 'f')
    ]

    assert graph.is_tour(xx) == False

    ##
    ## valid tour 1
    ##

    xx = [
        1.0, ## ('a', 'b')
        1.0, ## ('a', 'f')
        1.0, ## ('b', 'c')
        0.0, ## ('b', 'd')
        0.0, ## ('b', 'e')
        0.0, ## ('b', 'f')
        1.0, ## ('c', 'd')
        0.0, ## ('c', 'e')
        0.0, ## ('c', 'f')
        1.0, ## ('d', 'e')
        0.0, ## ('d', 'f')
        1.0, ## ('e', 'f')
    ]

    assert graph.is_tour(xx) == True
