#!/usr/bin/env python


import argparse
import re

class Node(object):
    def __init__(ident, coords, desc='N/A'):
        self.ident  = ident
        self.coords = coords
        self.desc   = desc


class Edge(object):
    def __init__(nodes, cost, desc='N/A'):
        self.nodes = nodes
        self.cost  = cost
        self.desc  = desc

class Tsp(object):
    def __init__(self, filename):
        self.filename = filename

        with open(self.filename, 'r') as fd:
            lines = [line for line in fd]

        self._parse_specification(lines)
        # self._parse_nodes(lines)


    def _parse_specification(self, lines):
        self.specs = {}

        def extract_name(self, line):
            match = re.match(r'^NAME : (.+)\s+$', line)
            if match:
                self.specs['name'] = match.group(1)
            return match

        def extract_type(self, line):
            match = re.match(r'^TYPE : (.+)\s+$', line)
            if match:
                self.specs['type'] = match.group(1)
            return match

        def extract_comment(self, line):
            match = re.match(r'^COMMENT : (.+)\s+$', line)
            if match:
                self.specs['comment'] = match.group(1)
            return match

        def extract_dimension(self, line):
            match = re.match(r'^DIMENSION : (.+)\s+$', line)
            if match:
                self.specs['dimension'] = int(match.group(1))
            return match

        def extract_capacity(self, line):
            match = re.match(r'^CAPACITY : (.+)\s+$', line)
            if match:
                self.specs['capacity'] = int(match.group(1))
            return match

        def extract_edge_weight_type(self, line):
            match = re.match(r'^EDGE_WEIGHT_TYPE : (.+)\s+$', line)
            if match:
                ewt = match.group(1)
                if ewt in ['EUC_2D', 'ATT']:
                    self.specs['edge_weight_type'] = ewt
                else:
                    raise StandardError('unknown/unsupported edge weight type: {}'.format(ewt))
            return match

        def extract_edge_weight_format(self, line):
            match = re.match(r'^EDGE_WEIGHT_FORMAT : (.+)\s+$', line)
            if match:
                self.specs['edge_weight_format'] = match.group(1)
            return match

        actions = [extract_name, extract_type, extract_comment, extract_dimension,
                   extract_edge_weight_type, edge_weight_format]

        for line in lines:
            for action in actions:
                if action(self, line):
                    break

# class Tour(object):
#     def __init__(nodes, edges):
#         validate(nodes, edges)
#         self.nodes = nodes
#         self.edges = edges

#     def validate(nodes, edges):
#         NN = len(nodes)
#         NE = len(edges)
#         if NN != NE:
#             raise StandardError('# edges ({}) != # nodes ({})'.format(NE, NN))

#         has_been_visited_by_node = {}
#         for edge in edges:
#             if has_been_visited_by_node[edge.node2()

class Solution(object):
    def __init__(nodes, edges):
        self.nodes = nodes
        self.edges = edges


    def plot(self):
        pass

def read_nodes(filename):
    with open(filename, 'r') as fd:
        for line in fd:
            pass


def main():
    args = parse_args()
    print('args = ({})'.format(args))
    tsp = Tsp(filename=args.tsp_filename)
    print('tsp = {}'.format(tsp.specs))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tsp',
        help='the filename containing the TSP definition',
        required=True,
        type=str,
        dest='tsp_filename',
    )
    parser.add_argument(
        '--tour',
        help='the filename containing the optimal tour associated with the TSP',
        type=str,
        dest='tour_filename',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
