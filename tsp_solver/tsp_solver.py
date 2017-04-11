from collections import deque
import gurobipy as grb
import math
import re
import random

random.seed(42)

from dataset import Dataset
# import subtour
from graph import Graph


def text_keys(text):
    ## http://stackoverflow.com/a/5967539/7002167
    def atoi(text):
        return int(text) if text.isdigit() else text

    return [atoi(ss) for ss in re.split('(\d+)', text)]


def tuple_keys(tpl):
    keys = []
    for item in tpl:
        keys.extend(text_keys(item))
    return keys

class TspBranchAndCut(object):

    def __init__(self, nodes, edges, cost_by_edge):
        self.nodes         = sorted(nodes, key=text_keys)
        self.edges         = sorted(edges, key=tuple_keys)
        self.cost_by_edge  = cost_by_edge

        self.node_by_idx = {idx: node for idx,node in enumerate(self.nodes)}
        self.idx_by_node = {node: idx for idx,node in enumerate(self.nodes)}

        self.edge_by_idx = {idx: edge for idx,edge in enumerate(self.edges)}
        self.idx_by_edge = {edge: idx for idx,edge in enumerate(self.edges)}

        self.queue         = deque()

        self.vars          = None
        self.best_cost     = 108160
        self.best_model    = None


    def solve(self):
        ##
        ## Formulate the initial model, which consists of:
        ##   - minimizing the sum of selected edges
        ##   - the degree requirements for each node
        ##   - the bounds on x_e (the decision variable)
        ##
        ##   min: sum_e(d_e*x_e)    for e in edges
        ##   st:  sum_e(x_e) = 2    e in edges(n) for n in nodes
        ##        0.0 <= x_e <= 1   for e in edges
        ##

        model          = grb.Model('tsp')
        xx             = model.addVars(self.edges, lb=0.0, ub=1.0, vtype=grb.GRB.CONTINUOUS, name='xx', obj=self.cost_by_edge)
        degree_constrs = model.addConstrs((xx.sum(node,'*') + xx.sum('*',node) == 2.0 for node in self.nodes), 'degree')
        model.update()

        model.setParam('OutputFlag', False)

        self.queue.append(grb.Model.copy(model))

        while len(self.queue) != 0:
            print('popping from queue')
            if False: #self.best_cost is None:
                model = self.queue.pop()
            # elif len(self.queue) >= 250:
            #     self.queue.rotate(random.randint(1,len(self.queue)))
            #     model = self.queue.pop()
            else:
                model = self.queue.popleft()

            while True:
                print('optimizing: bc: {} ql: {} nc: {}'.format(self.best_cost, len(self.queue), len(model.getConstrs())))
                model.update()
                model.optimize()
                # print('status: {}'.format(model.status))

                if self.solution_is_infeasible(model):
                    print('  infeasible')
                    break
                if not self.solution_can_become_new_best(model):
                    print('  cannot become best')
                    break

                if self.solution_is_tour(model):
                    print('  tour')
                    if self.solution_is_new_best(model):
                        print('  new best')
                        self.update_best(model)
                    break

                print('adding constraints')
                constraints_were_added = False
                constraints_were_added = self.add_integral_subtour_constraints(model)    | constraints_were_added
                constraints_were_added = self.add_nonintegral_subtour_constraints(model) | constraints_were_added
                constraints_were_added = self.add_comb_constraints(model)                | constraints_were_added

                if constraints_were_added:
                    print('  constraints were added')
                    continue

                if not self.solution_is_integral(model):
                    print('adding branches')
                    models = self.create_branch_models(model)
                    if len(models) == 0:
                        raise StandardError('no branch models could be found')
                    self.queue.extend(models)
                    break


    def solution_is_infeasible(self, model):
        return grb.GRB.INFEASIBLE == model.status


    def solution_can_become_new_best(self, model):
        if grb.GRB.OPTIMAL != model.status:
            return False
        if self.best_cost is None:
            return True
        if model.getAttr('ObjVal') < self.best_cost:
            return True
        return False


    def solution_is_tour(self, model):
        graph, xx = self.convert_model(model)
        return graph.is_tour(solution=xx)


    def convert_model(self, model):
        solution = model.getAttr('X')
        weight_by_edge = {edge: solution[idx] for idx,edge in enumerate(self.edges)}
        graph = Graph(nodes=self.nodes, edges=self.edges, weight_by_edge=weight_by_edge)

        return (graph, solution)


    def add_integral_subtour_constraints(self, model):
        if not self.solution_is_integral(model):
            return False

        graph, xx = self.convert_model(model)
        connected_component_nodes, connected_component_edges = graph.binary_connected_components(solution=xx)

        if len(connected_component_nodes) == 1:
            return False

        # import pdb; pdb.set_trace()

        models = [mm for mm in self.queue]
        models.append(model)

        for model in models:
            mvars = model.getVars()

            for cc in connected_component_nodes:
                cut_edges = graph.get_cut_edges(nodes=cc)
                var_idxs = sorted([self.idx_by_edge[edge] for edge in cut_edges])
                cvars = [mvars[idx] for idx in var_idxs]

                # import pdb; pdb.set_trace()
                model.addConstr(grb.quicksum(cvars) >= 2.0, 'subtour-integral')
                # model.update()

        return True

    def add_nonintegral_subtour_constraints(self, model):
        if self.solution_is_integral(model):
            return False

        graph, xx = self.convert_model(model)
        min_cut_value, min_cut_nodes = graph.find_min_cut()

        if min_cut_value >= 2.0 - 1e-8:
            return False

        print('min_cut_value = {:+1.16e}'.format(min_cut_value))
        print('min_cut_nodes = [{}] {}'.format(len(min_cut_nodes), min_cut_nodes))

        models = [mm for mm in self.queue]
        models.append(model)

        for model in models:
            mvars = model.getVars()

            cut_edges = graph.get_cut_edges(nodes=min_cut_nodes)
            print('cut_edges = [{}] {}'.format(len(cut_edges), cut_edges))
            cut_weight = 0
            for edge in cut_edges:
                cut_weight += graph.weight_by_edge[edge]
            print('cut_weight = {}'.format(cut_weight))

            var_idxs = sorted([self.idx_by_edge[edge] for edge in cut_edges])
            cvars = [mvars[idx] for idx in var_idxs]

            # import pdb; pdb.set_trace()
            model.addConstr(grb.quicksum(cvars) >= 2.0, 'subtour-nonintegral')
            # model.update()

        return True


    def add_comb_constraints(self, model):
        graph, xx = self.convert_model(model)

        ##
        # Find the connected components of the G12 graph, which
        ## is G with the edges with decision values 1.0,0.0 removed.
        ##

        g12_edges = [edge for edge in graph.edges if xx[graph.idx_by_edge[edge]] not in [1.0,0.0]]
        # print('g12_edges = {}'.format(g12_edges))
        weight_by_edge = {edge: 0.0 for edge in g12_edges}
        g12 = Graph(nodes=graph.nodes, edges=g12_edges, weight_by_edge=weight_by_edge)
        # print('g12.edges_by_node = {}'.format(g12.edges_by_node))
        ccs_nodes, ccs_edges = g12.connected_components()

        ##
        ##
        ##

        constraints_were_added = False

        for cc_idx,cc_nodes in enumerate(ccs_nodes):
            cc_cut_edges = graph.get_cut_edges(nodes=cc_nodes)
            one_edges = [edge for edge in cc_cut_edges if xx[graph.idx_by_edge[edge]] == 1.0]
            if len(one_edges) % 2 == 1:
                while True:
                    external_node_intersection_count = {node: 0 for node in nodes}
                    for edge in one_edges:
                        for node in edge:
                            if node not in nodes:
                                external_node_intersection_count[node] += 1

                    multiply_intersected_nodes = [node for node,count in external_node_intersection_count.items() if count > 1]

                    if len(multiply_intersected_nodes) == 0:
                        break

                    print('comb - multiply intersected nodes: {}'.format(multiply_intersected_nodes))
                    for node in multiply_intersected_nodes:
                        cc_nodes.append(node)
                    cc_cut_edges = graph.get_cut_edges(nodes=cc_nodes)
                    one_edges = [edge for edge in cc_cut_edges if xx[graph.idx_by_edge[edge]] == 1.0]

                cc_nodes  = sorted(set(cc_nodes), key=text_keys)
                one_edges = sorted(set(one_edges), key=tuple_keys)

                if len(one_edges) == 1:
                    print('comb - subtour')

                    handle_cut_edges = sorted(set(graph.get_cut_edges(nodes=cc_nodes)), key=tuple_keys)
                    handle_var_idxs  = sorted([self.idx_by_edge[edge] for edge in handle_cut_edges])

                    models = [mm for mm in self.queue]
                    models.append(model)

                    for midx,model in enumerate(models):

                        mvars = model.getVars()

                        handle_vars = [mvars[idx] for idx in handle_var_idxs]

                        model.addConstr(grb.quicksum(handle_vars) >= 2.0, 'comb-subtour')
                        # model.update()

                    constraints_were_added = True

                else:
                    print('comb - blossom H={} T={}'.format(cc_nodes,one_edges))

                    handle_cut_edges = sorted(set(graph.get_cut_edges(nodes=cc_nodes)), key=tuple_keys)
                    tooth_cut_edges  = [graph.get_cut_edges(nodes=tooth_nodes) for tooth_nodes in one_edges]
                    tooth_cut_edges  = sorted([item for sublist in tooth_cut_edges for item in sublist], key=tuple_keys)

                    # print('  handle_cut_edges = {}'.format(handle_cut_edges))
                    # print('  tooth_cut_edges  = {}'.format(tooth_cut_edges))

                    handle_var_idxs = sorted([self.idx_by_edge[edge] for edge in handle_cut_edges])
                    tooth_var_idxs  = sorted([self.idx_by_edge[edge] for edge in tooth_cut_edges])

                    models = [mm for mm in self.queue]
                    models.append(model)

                    for midx,model in enumerate(models):

                        mvars = model.getVars()

                        handle_vars = [mvars[idx] for idx in handle_var_idxs]
                        tooth_vars  = [mvars[idx] for idx in tooth_var_idxs]

                        # import pdb; pdb.set_trace()
                        expr = grb.quicksum(handle_vars) + grb.quicksum(tooth_vars) >= 3*len(one_edges) + 1

                        # if midx==0:
                        #     print(expr)
                        model.addConstr(expr, 'comb')
                        # model.update()

                    constraints_were_added = True

        return constraints_were_added


    def solution_is_integral(self, model):
        if grb.GRB.OPTIMAL != model.status:
            return False

        for mvar in model.getVars():
            val = mvar.getAttr('X')
            if abs(val - int(val)) > 1e-8:
                return False

        return True


    def solution_is_new_best(self, model):
        if grb.GRB.OPTIMAL != model.status:
            return False
        return (self.best_cost is None) or (model.getAttr('ObjVal') < self.best_cost)


    def update_best(self, model):
        if grb.GRB.OPTIMAL != model.status:
            raise StandardError('model was not solved to optimality')
        self.best_cost  = model.getAttr('ObjVal')
        self.best_model = model


    def create_branch_models(self, model):
        if grb.GRB.OPTIMAL != model.status:
            raise StandardError('model was not solved to optimality')

        for mvar in model.getVars():
            val = mvar.getAttr('X')
            if abs(val - int(val)) != 0.0:
                model1 = grb.Model.copy(model)
                m1var  = model1.getVarByName(mvar.getAttr('VarName'))
                model1.addConstr(m1var == math.floor(val))
                model1.update()

                model2 = grb.Model.copy(model)
                m2var  = model2.getVarByName(mvar.getAttr('VarName'))
                model2.addConstr(m2var == math.ceil(val))
                model2.update()

                return (model1, model2)

        return ()


    def model_to_str(self, model, indent=0):
        model.write('tmp.lp')
        lines = deque()
        with open('tmp.lp', 'r') as fd:
            lines.extend(' '*indent + line for line in fd)
        lines.popleft()
        lines.popleft()

        if grb.GRB.OPTIMAL == model.status:
            for mvar in model.getVars():
                lines.append(' '*indent + '{} = {:+1.5e}\n'.format(mvar.getAttr('VarName'), mvar.getAttr('X')))

        return ''.join(lines)


if __name__ == '__main__':
    import sys

    ##
    ## Read the dataset
    ##

    filename         = sys.argv[1]
    dataset          = Dataset(filename)
    edges            = dataset.edges
    nodes            = dataset.nodes
    distance_by_edge = dataset.distance_by_edge

    ##
    ## Solve the problem.
    ##

    bc = TspBranchAndCut(nodes=nodes, edges=edges, cost_by_edge=distance_by_edge)
    bc.solve()
    print('BEST COST: {}'.format(bc.best_cost))
