#!/usr/bin/env python

import copy
import math
import random

def cost(nodes, edges):
    cost = 0.
    for edge in edges:
        n1 = nodes[edge[0]]
        n2 = nodes[edge[1]]
        cost += ( (n1[1] - n2[1])**2 + (n1[2] - n2[2])**2 - edge[2]**2 )**2
    return cost

def adjust(nodes, edges, dist_by_edge):
    def cost():
        cost = 0.0
        for edge in edges:
            x1 = coords_by_node[edge[0]][0]
            y1 = coords_by_node[edge[0]][1]
            x2 = coords_by_node[edge[1]][0]
            y2 = coords_by_node[edge[1]][1]

            dist12 = dist_by_edge[edge]

            kk = ((x1 - x2)**2 + (y1 - y2)**2 - dist12**2)**2
            cost += kk

        return cost

    def normalize(coords_by_node):
        min_x = min([coords[0] for node,coords in coords_by_node.items()])
        max_x = max([coords[0] for node,coords in coords_by_node.items()])
        min_y = min([coords[1] for node,coords in coords_by_node.items()])
        max_y = max([coords[1] for node,coords in coords_by_node.items()])

        dx = max_x - min_x
        dy = max_y - min_y

        new_coords_by_node = {node: [(coords[0] - min_x)/dx, (coords[1] - min_y)/dy] for node,coords in coords_by_node.items()}

        return new_coords_by_node

    coords_by_node = {node: [idx, random.randint(-10, +10)] for idx,node in enumerate(nodes) }

    old_cost = cost()
    new_cost = old_cost

    step = 1e-10
    iter_num = 0
    while True:
        iter_num += 1

        old_cost = new_cost
        new_cost = cost()
        ratio = abs(old_cost - new_cost)/new_cost

        print('iter_num = {:03d} oc: {:+1.6e} nc: {:+1.6e} r: {:+1.6e}'.format(iter_num, old_cost, new_cost, ratio))

        # for idx,node in enumerate(nodes):
        #     print('node[{:03d}]: {:+1.4e} {:+1.4e}'.format(idx, coords_by_node[node][0], coords_by_node[node][1]))

        if iter_num > 1:
            if abs(old_cost - new_cost)/new_cost < 1e-6:
                break

        new_coords_by_node = copy.deepcopy(coords_by_node)
        for n1 in nodes:
            dcdx = 0.
            dcdy = 0.

            x1 = coords_by_node[n1][0]
            y1 = coords_by_node[n1][1]

            for edge in edges:
                if edge[0] == n1:
                    n2 = edge[1]
                elif edge[1] == n1:
                    n2 = edge[0]
                else:
                    continue

                x2 = coords_by_node[n2][0]
                y2 = coords_by_node[n2][1]
                dist12 = dist_by_edge[edge]

                kk = (x1 - x2)**2 + (y1 - y2)**2 - dist12**2
                # print('kk = {:+1.5e}'.format(kk))
                dcdx += 2*kk*(x1 - x2)
                dcdy += 2*kk*(y1 - y2)

            # print('dcdx = {:+1.5e} dcdy = {:+1.5e}'.format(dcdx, dcdy))
            new_coords_by_node[n1][0] = coords_by_node[n1][0] - step*dcdx
            new_coords_by_node[n1][1] = coords_by_node[n1][1] - step*dcdy

            # new_coords_by_node = normalize(new_coords_by_node)

        coords_by_node = new_coords_by_node

    coords_by_node = normalize(coords_by_node)

    return coords_by_node


def main():
    from dataset import Dataset

    with open('../datasets/class/att48.txt') as fd:
        dataset = Dataset(fd)

    coords_by_node = adjust(nodes=dataset.nodes, edges=dataset.edges, dist_by_edge=dataset.distance_by_edge)
    for node in dataset.nodes:
        print('{:3s} {:1.3e} {:1.3e}'.format(node, coords_by_node[node][0], coords_by_node[node][1]))


if __name__ == '__main__':
    main()
