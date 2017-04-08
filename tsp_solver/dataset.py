from collections import deque
import re

class Dataset(object):

    def __init__(self, filename):
        self.filename = filename

        ##
        ## Open and read the file, removing extra whitespaces,
        ## blank lines, and comments.
        ##

        lines = deque()
        with open(filename, 'r') as fd:
            for line in fd:
                ss = line.strip()
                if ss != '' and not re.match(r'^#', line):
                    lines.append(ss)

        ##
        ## Extract the header information from the first line.
        ##

        if len(lines) == 0:
            raise StandardError('no header line found ({})'.format(filename))

        header_line = lines.popleft()
        match_obj = re.match(r'^\s*(\d+)\s+(\d+)\s*$', header_line)
        if match_obj:
            num_nodes = int(match_obj.group(1))
            num_edges = int(match_obj.group(2))
        else:
            raise StandardError('invalid header line: ({})'.format(header_line))

        ##
        ## Some of the datasets contain a random number as
        ## their last line.  If that's the case here, remove it.
        ##

        match_obj = re.match(r'^\d+$', lines[-1])
        if match_obj:
            lines.pop()

        ##
        ## All remaining lines should be edge-distance definitions.
        ##

        nodes            = set()
        distance_by_edge = {}
        while len(lines) != 0:
            line = lines.popleft()

            match_obj = re.match(r'^(\d+)\s+(\d+)\s+(\d+)$', line)
            if match_obj:
                try:
                    node1     = match_obj.group(1)
                    node2     = match_obj.group(2)
                    distance  = int(match_obj.group(3))
                except ValueError:
                    raise StandardError('invalid edge value: ({})'.format(line))

                if node1 == node2:
                    raise StandardError('no loops allowed: ({})'.format(line))

                nodes.add(node1)
                nodes.add(node2)

                edge = (node1,node2)

                if edge in distance_by_edge:
                    raise StandardError('duplicate edge found: {}'.format(line))
                distance_by_edge[(node1,node2)] = distance
            else:
                raise StandardError('invalid edge line: ({})'.format(line))

        if len(nodes) != num_nodes:
            raise StandardError('file {} did not contain the correct number of nodes (act {} != {} exp)'.format(filename, len(nodes), num_nodes))
        if len(distance_by_edge) != num_edges:
            raise StandardError('file {} did not contain the correct number of nodes (act {} != {} exp)'.format(filename, len(distance_by_edge), num_edges))

        self.nodes            = sorted(nodes)
        self.edges            = sorted(distance_by_edge.keys())
        self.distance_by_edge = distance_by_edge


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    print('filename = {}'.format(filename))
    ds = Dataset(filename=filename)