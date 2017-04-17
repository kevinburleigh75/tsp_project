#!/usr/bin/env python

import re

def convert(istream, ostream):
    nodes = []

    for line in istream:
        match_obj = re.match(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s*$', line)
        if match_obj:
            node = int(match_obj.group(1)) - 1
            xx   = float(match_obj.group(2))
            yy   = float(match_obj.group(3))

            nodes.append( (node, xx, yy) )

    _, min_xx, _ = min(nodes, key=lambda elem: elem[1])
    _, max_xx, _ = max(nodes, key=lambda elem: elem[1])
    xx_range = max_xx - min_xx

    _, _, min_yy = min(nodes, key=lambda elem: elem[2])
    _, _, max_yy = max(nodes, key=lambda elem: elem[2])
    yy_range = max_yy - min_yy

    for node,xx,yy in nodes:
        new_xx = (xx - min_xx)/xx_range
        new_yy = (yy - min_yy)/yy_range
        ostream.write('{} {} {}\n'.format(node, new_xx, new_yy))


if __name__ == '__main__':
    import sys
    convert(sys.stdin, sys.stdout)
