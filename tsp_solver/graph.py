from collections import deque
import copy

## https://www.safaribooksonline.com/library/view/python-cookbook-2nd/0596007973/ch04s07.html
def list_or_tuple(x):
    return isinstance(x, (list, tuple))

def flatten(sequence, to_expand=list_or_tuple):
    for item in sequence:
        if to_expand(item):
            for subitem in flatten(item, to_expand):
                yield subitem
        else:
            yield item

class Graph(object):

    def __init__(self, nodes, edges, weight_by_edge):
        self.nodes          = list(nodes)
        self.edges          = list(edges)
        self.weight_by_edge = copy.copy(weight_by_edge)

        for edge in self.edges:
            if edge not in weight_by_edge:
                raise ValueError('missing edge weight for edge {}'.format(edge))

        self.num_nodes = len(self.nodes)
        self.num_edges = len(self.edges)

        if self.num_nodes == 0:
            raise ValueError('zero nodes does not a graph make...')

        self.edge_set  = set(self.edges)
        self.node_set  = set(self.nodes)

        if len(self.edge_set) != len(self.edges):
            raise ValueError('duplicate edges detected')
        if len(self.node_set) != len(self.nodes):
            raise ValueError('duplicate nodes detected')

        for edge in self.edges:
            (node1, node2) = edge
            if (node2, node1) in self.edge_set:
                raise ValueError('flipped duplicate edge deteced: {}'.format(edge))

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


    def __repr__(self):
        return '<Graph nodes: {} weights_by_edge: {}>'.format(self.nodes, self.weight_by_edge)


    def num_nodes(self):
        return self.num_nodes


    def edge_by_nodes(self, node1, node2):
        for edge in [(node1,node2),(node2,node1)]:
            if edge in self.edge_set:
                return edge
        raise ValueError('there is no edge between nodes {} and {}'.format(node1, node2))


    def weight_by_nodes(self, node1, node2):
        return self.weight_by_edge[self.edge_by_nodes(node1, node2)]


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
        weight_by_edge = {edge: self.weight_by_edge[edge] for edge in binary_one_edges}
        binary_one_graph = Graph(nodes=self.nodes, edges=binary_one_edges, weight_by_edge=weight_by_edge)

        ccs_nodes, ccs_edges = binary_one_graph.connected_components()
        return (ccs_nodes, ccs_edges)


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


    def find_min_cut(self):
        if self.num_nodes < 2:
            raise StandardError('cannot cut a graph with {} nodes'.format(self.num_nodes))

        merged_graph = Graph(nodes=self.nodes, edges=self.edges, weight_by_edge=self.weight_by_edge)

        min_cut_value = float("inf")
        min_cut_nodes = []

        all_cuts = []

        while merged_graph.num_nodes > 1:
            # print('-'*20)
            # print('  min_cut_value = {}'.format(min_cut_value))
            # print('  min_cut_nodes = {}'.format(sorted(flatten(min_cut_nodes, list_or_tuple))))
            # print('  non_cut_nodes = {}'.format(sorted(list(set(flatten(merged_graph.nodes,list_or_tuple))-set(flatten(min_cut_nodes,list_or_tuple))))))
            # print('merged_graph = {}'.format(merged_graph))
            # print('merged_graph,num_nodes = {}'.format(merged_graph.num_nodes))

            node_set = [merged_graph.nodes[0]]

            while True:
                next_node = merged_graph._find_next_connected_node(target_nodes=node_set)
                if next_node is None:
                    break
                node_set.append(next_node)

            # print('node set = {}'.format(node_set))

            penultimate_node = node_set[-2]
            last_node        = node_set[-1]

            current_cut_value = 0
            current_cut_nodes = [last_node]
            for other_node in merged_graph.nodes_by_node[last_node]:
                current_cut_value += merged_graph.weight_by_nodes(last_node, other_node)
                # current_cut_nodes.append(other_node)

            merged_graph = merged_graph._merge_nodes(node1=penultimate_node, node2=last_node)

            all_cuts.append((current_cut_value, [item for item in flatten(current_cut_nodes, list_or_tuple)]))

            if current_cut_value < min_cut_value:
                min_cut_value = current_cut_value
                min_cut_nodes = current_cut_nodes

        all_cuts = sorted(all_cuts, key=lambda x: x[0])

        # min_cut_nodes = [item for item in flatten(min_cut_nodes, list_or_tuple)]
        # return min_cut_value, min_cut_nodes

        return all_cuts


    def _find_next_connected_node(self, target_nodes):
        """
        Finds the tightly connected node in V not in A
        Input: graph with node 1 and node 2 keys and weight values,
        node_set: set representing A from min cut algorithm
        Output: node that is tightly connected not in A
        """

        ## convert target_nodes to a set to ensure fast lookup
        target_nodes = set(target_nodes)

        max_weight = -float('inf')
        best_node  = None

        for node1 in set(self.nodes) - set(target_nodes):
            current_weight = 0
            for node2 in target_nodes:
                if node2 in self.nodes_by_node[node1]:
                    current_weight += self.weight_by_nodes(node1, node2)
            if current_weight > max_weight:
                best_node  = node1
                max_weight = current_weight

        return best_node


    def _merge_nodes(self, node1, node2):
        """
        merges string type nodes t and s and returns a modified copy of
        the old graph with updated edge weights
        """

        ##
        ## Create a new node and edge lists, but exclude
        ## anything affected by node1 or node2.
        ##

        new_nodes          = list(set(self.nodes) - set([node1, node2]))
        new_edges          = [edge for edge in self.edges if len(set(edge) - set([node1,node2])) == 2]
        new_weight_by_edge = {edge: weight for edge,weight in self.weight_by_edge.items() if len(set(edge) - set([node1,node2])) == 2}

        ##
        ## Create the new node and its merged edges.
        ##

        # new_node = '--'.join([str(node1), str(node2)])
        new_node = (node1, node2)
        new_nodes.append(new_node)

        for other_node in self.nodes_by_node[node1]:
            if other_node == node2:
                continue
            if node2 in self.nodes_by_node[other_node]:
                new_edge = (new_node,other_node)
                new_edges.append(new_edge)
                new_weight_by_edge[new_edge] = self.weight_by_nodes(other_node, node1) + self.weight_by_nodes(other_node, node2)
            else:
                new_edge = (new_node,other_node)
                new_edges.append(new_edge)
                new_weight_by_edge[new_edge] = self.weight_by_nodes(node1, other_node)

        for other_node in self.nodes_by_node[node2]:
            if other_node == node1:
                continue
            if node1 not in self.nodes_by_node[other_node]:
                new_edge = (new_node,other_node)
                new_edges.append(new_edge)
                new_weight_by_edge[new_edge] = self.weight_by_nodes(node2, other_node)

        # print('new nodes: {}'.format(new_nodes))
        # print('new edges: {}'.format(new_edges))
        # print('new wbe:   {}'.format(new_weight_by_edge))
        merged_graph = Graph(
            nodes          = new_nodes,
            edges          = new_edges,
            weight_by_edge = new_weight_by_edge,
        )

        return merged_graph


if __name__ == '__main__':

    ##############################################
    ##
    ## is_tour
    ##
    ##############################################

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

    weight_by_edge = {edge: 1.0 for edge in edges}

    graph = Graph(nodes=nodes, edges=edges, weight_by_edge=weight_by_edge)

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

    ##############################################
    ##
    ## find_min_cut
    ##
    ##############################################

    nodes = ['1', '2', '3', '4', '5', '6', '7', '8']
    edges = [
        ('1', '2'),
        ('1', '5'),
        ('2', '3'),
        ('2', '5'),
        ('2', '6'),
        ('3', '4'),
        ('3', '7'),
        ('4', '7'),
        ('4', '8'),
        ('5', '6'),
        ('6', '7'),
        ('7', '8'),
    ]
    weight_by_edge = {
        ('1', '2'): 2.0,
        ('1', '5'): 3.0,
        ('2', '3'): 3.0,
        # ('2', '3'): 0.0,
        ('2', '5'): 2.0,
        ('2', '6'): 2.0,
        ('3', '4'): 4.0,
        ('3', '7'): 2.0,
        ('4', '7'): 2.0,
        ('4', '8'): 2.0,
        ('5', '6'): 3.0,
        ('6', '7'): 1.0,
        # ('6', '7'): 0.0,
        ('7', '8'): 3.0,
    }

    graph = Graph(nodes=nodes, edges=edges, weight_by_edge=weight_by_edge)
    print('graph = {}'.format(graph))

    (min_cut_value, min_cut_nodes) = graph.find_min_cut()
    print('min_cut_value = {} min_cut_nodes = {}'.format(min_cut_value, min_cut_nodes))
