#!/usr/bin/env python

from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import networkx as nx

Node = namedtuple('Node', 'name coords')
Edge = namedtuple('Edge', 'node1 node2 cost')

def plot_solution(nodes, edges, solution_str, title):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)

    cmap  = plt.cm.gist_rainbow
    cNorm = colors.Normalize(vmin=0, vmax=1)

    pos_by_node = {node: node.coords for node in nodes}

    edges_by_code = {code: set() for code in ['1', 'N', '0', 'Z', '5', 'L', 'H']}
    for idx,edge in enumerate(edges):
        code = solution_str[idx]
        edges_by_code[code].add( (edge.node1,edge.node2) )

    plt.cla()
    axes = plt.gca()
    axes.set_xlim([-0.1,1.1])
    axes.set_ylim([-0.1,1.1])

    nx.draw_networkx_nodes(graph,
        ax         = axes,
        pos        = pos_by_node,
        node_color = 'k',
        node_size  = 10,
    )

    nx.draw_networkx_edges(graph,
        ax         = axes,
        pos        = pos_by_node,
        edgelist   = edges_by_code['1'],
        edge_color = 'k',
    )

    nx.draw_networkx_edges(graph,
        ax         = axes,
        pos        = pos_by_node,
        edgelist   = edges_by_code['N'],
        edge_color = 'b',
    )

    nx.draw_networkx_edges(graph,
        ax         = axes,
        pos        = pos_by_node,
        edgelist   = edges_by_code['5'],
        edge_color = 'm',
    )

    nx.draw_networkx_edges(graph,
        ax         = axes,
        pos        = pos_by_node,
        edgelist   = edges_by_code['Z'],
        edge_color = 'c',
    )

    nx.draw_networkx_edges(graph,
        ax         = axes,
        pos        = pos_by_node,
        edgelist   = edges_by_code['L'],
        edge_color = 'g',
    )

    nx.draw_networkx_edges(graph,
        ax         = axes,
        pos        = pos_by_node,
        edgelist   = edges_by_code['H'],
        edge_color = 'r',
    )

    label_pos_by_node = {node: (node.coords[0]-0.03, node.coords[1]) for node in nodes}
    label_by_node     = {node: node.name for node in nodes}
    nx.draw_networkx_labels(graph,
        ax         = axes,
        pos        = label_pos_by_node,
        labels     = label_by_node,
        font_size  = 10,
    )

    plt.title(title)

    #     if code == '1':
    #         color_by_edge[edge] = 'k'
    #     elif code == 'N':
    #         color_by_edge[edge] = 'b'
    #     elif code == 'Z':
    #         color_by_edge[edge] = 'm'
    #     elif code == '5':
    #         color_by_edge[edge] = 'r'
    #     elif code == '-':
    #         color_by_edge[edge] = 'c'
    #     else:
    #         raise StandardError('unknown code: {}'.format())

    # nx.draw(graph,
    #     cmap       = cmap,
    #     pos        = pos_by_node,
    #     node_color = 'k',
    #     node_size  = 10,
    #     edge_color = color_by_edge,
    # )
    # plt.pause(0.025)
    plt.show()


if __name__ == '__main__':
    import re
    import sys

    dataset_name   = sys.argv[1]
    soln_id_str    = sys.argv[2]

    node_by_name = {}
    with open(dataset_name+'.coords', 'r') as fd:
        for line in fd:
            match_obj = re.match(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s*$', line)
            if match_obj:
                name = match_obj.group(1)
                xx   = float(match_obj.group(2))
                yy   = float(match_obj.group(3))
                node_by_name[name] = Node(name=name, coords=(xx,yy))
            else:
                raise StandardError('invalid line: {}'.format(line))
    nodes = sorted(node_by_name.values())

    edges    = []
    soln_str = None
    with open(dataset_name+'.info', 'r') as fd:
        for line in fd:
            match_obj = re.match(r'^.*?NODES:\s+(.*)\s*$', line)
            if match_obj:
                names = match_obj.group(1).split()
                if len(names) != len(node_by_name.keys()):
                    raise StandardError('file mismatch {} != {}'.format(len(names), len(node_by_name.keys())))
                for name in names:
                    if name not in node_by_name:
                        raise StandardError('missing node: {}'.format(name))

            match_obj = re.match(r'^.*?EDGES:\s+(.*)\s*$', line)
            if match_obj:
                edge_strs = re.findall(r'\((\S+)\s+(\S+)\)', match_obj.group(1))
                for name1, name2 in edge_strs:
                    if (name1 not in node_by_name) or (name2 not in node_by_name):
                        raise StandardError('invalid edge: {} {}'.format(name1, name2))
                    edges.append( Edge(node1=node_by_name[name1], node2=node_by_name[name2], cost=1.0) )

            match_obj = re.search(soln_id_str + r'.*\s(\S+)\s*$', line)
            if match_obj:
                soln_str = match_obj.group(1)

    if soln_str is None:
        raise StandardError('could not find solution for: {}'.format(soln_id_str))

    title = '{} ({})'.format(dataset_name, soln_id_str)
    plot_solution(nodes=nodes, edges=edges, solution_str=soln_str, title=title)

