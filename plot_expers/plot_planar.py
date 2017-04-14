#!/usr/bin/env python

import math
import random

def cost(nodes, edges):
    cost = 0.
    for edge in edges:
        n1 = nodes[edge[0]]
        n2 = nodes[edge[1]]
        cost += ( (n1[1] - n2[1])**2 + (n1[2] - n2[2])**2 - edge[2]**2 )**2
    return cost

def adjust(nodes, edges):
    step = 1e-2
    for n1 in nodes:
        dcdx = 0.
        dcdy = 0.
        for edge in edges:
            if edge[0] == n1[0]:
                n2 = nodes[edge[1]]
            elif edge[1] == n1[0]:
                n2 = nodes[edge[0]]
            else:
                continue

            kk = (n1[1] - n2[1])**2 + (n1[2] - n2[2])**2 - edge[2]**2
            # print('kk = {:+1.5e}'.format(kk))
            dcdx += kk*(n1[1] - n2[1])
            dcdy += kk*(n1[2] - n2[2])

        # print('dcdx = {:+1.5e} dcdy = {:+1.5e}'.format(dcdx, dcdy))
        n1[1] += -step*dcdx
        n1[2] += -step*dcdy


def main():
    edges = [
        (0, 1, 1),
        (0, 2, 1),
        (0, 3, math.sqrt(2)),
        (0, 4, 2),
        (0, 5, math.sqrt(5)),
        (1, 2, math.sqrt(2)),
        (1, 3, 1),
        (1, 4, math.sqrt(5)),
        (1, 5, 2),
        (2, 3, 1),
        (2, 4, 1),
        (2, 5, math.sqrt(2)),
        (3, 4, math.sqrt(2)),
        (3, 5, 1),
        (4, 5, 1),
    ]

    nodes = []
    for idx in xrange(6):
        nodes.append([idx, random.normalvariate(0,1), random.normalvariate(0,1)])

    cc = cost(nodes, edges)
    print('cc = {:+1.5e}'.format(cc))
    while cc > 1e-6:
        adjust(nodes, edges)
        cc = cost(nodes, edges)
    print('cc = {:+1.5e}'.format(cc))

    for node in nodes:
        print('node[{:02d}] = ({:+1.2e}, {:+1.2e}'.format(node[0], node[1], node[2]))


if __name__ == '__main__':
    main()
