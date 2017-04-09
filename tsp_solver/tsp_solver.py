from collections import deque
import gurobipy as grb
import math
import re

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
        self.best_cost     = None
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
            print('='*40)
            print('='*40)
            model = self.queue.popleft()

            model.optimize()
            print('best_cost = {}'.format(self.best_cost))
            # print('='*40)
            # print(self.model_to_str(model, indent=2))
            # print('='*40)

            if self.solution_is_infeasible(model):
                print('  INFEASIBLE')
                continue
            if not self.solution_can_become_new_best(model):
                print('  CANNOT BECOME BEST')
                continue

            print('  ADDING CUTS (if possible)')
            while True:
                if self.solution_is_tour(model):
                    break

                if self.solution_is_integral(model):
                    # print('    SOLUTION IS INTEGRAL')
                    if not self.add_subtour_constraints(model):
                        # print('      NO CUTS WERE ADDED')
                        break
                else:
                    break

                print('    optimizing - {}'.format(len(model.getConstrs())))
                model.optimize()
                if self.solution_is_infeasible(model):
                    break

                print('    status    = {}'.format(model.status))
                print('    best_cost = {}'.format(self.best_cost))
                print('    cur_cost  = {}'.format(model.getAttr('ObjVal')))


            if self.solution_is_infeasible(model):
                print('  INFEASIBLE')
                continue
            if not self.solution_can_become_new_best(model):
                print('  CANNOT BECOME BEST')
                continue

            if self.solution_is_tour(model):
                print('    SOLUTION IS TOUR')
                if self.solution_is_new_best(model):
                    print('      NEW BEST - FATHOMED')
                    self.update_best(model)
                else:
                    print('      NOT A NEW BEST - FATHOMED')
            elif not self.solution_is_integral(model):
                print('    BRANCHING ON NON-INTEGER X')
                models = self.create_branch_models(model)
                if len(models) == 0:
                    raise StandardError('no branch models could be found')
                self.queue.extend(models)
            else:
                raise StandardError('integral non-tour solution - coding error!')


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
        graph = Graph(nodes=self.nodes, edges=self.edges, cost_by_edge=self.cost_by_edge)
        solution = model.getAttr('X')

        return (graph, solution)


    def add_subtour_constraints(self, model):
        graph, xx = self.convert_model(model)
        connected_component_nodes, connected_component_edges = graph.binary_connected_components(solution=xx)

        if len(connected_component_nodes) == 1:
            return False

        # import pdb; pdb.set_trace()

        mvars = model.getVars()

        for cc in connected_component_nodes:
            cut_edges = graph.get_cut_edges(nodes=cc)
            var_idxs = sorted([self.idx_by_edge[edge] for edge in cut_edges])
            cvars = [mvars[idx] for idx in var_idxs]

            # import pdb; pdb.set_trace()
            model.addConstr(grb.quicksum(cvars) >= 2.0, 'subtour')

        model.update()

        return True

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

    # bc = BranchAndCut(initial_model=model)
    bc = TspBranchAndCut(nodes=nodes, edges=edges, cost_by_edge=distance_by_edge)
    bc.solve()
    print('BEST COST: {}'.format(bc.best_cost))

    # import pdb; pdb.set_trace()
    # print('hello')


# def print_info(model):
#     # model.write('model.mps')
#     print('=' * 40)
#     print('INFO:')
#     if model.getAttr('ModelSense') > 0:
#         print('  min:')
#     else:
#         print('  max:')
#     print('    {}'.format(model.getObjective()))

#     print('  subject to:')
#     mvars = model.getVars()

#     for cc in model.getConstrs():
#         rr = model.getRow(cc)
#         ss = '    '
#         for idx in range(rr.size()):
#             coeff = rr.getCoeff(idx)
#             mvar  = rr.getVar(idx)
#             ss += '{:+1.3e} * {}  '.format(coeff, mvar.getAttr('VarName'))
#         ss += '{}= '.format(cc.getAttr('Sense'))
#         ss += '{}'.format(cc.getAttr('RHS'))
#         print(ss)

#     if grb.GRB.OPTIMAL == model.status:
#         print('  optimal solution:')
#         for var in model.getVars():
#             print('    {:5.5s} {:+1.5e}'.format(var.getAttr('VarName'), var.getAttr('X')))
#     else:
#         print('  no optimum found')
#     print('=' * 40)

# mm = grb.Model()

# vv = [1, 2]

# cost = {
#     1: -1.0,
#     2: -1.0,
# }

# xx = mm.addVars(vv, name='x', obj=cost, vtype=grb.GRB.CONTINUOUS, lb=0.0)
# mm.addConstr(xx[1] + 2.0/3.0*xx[2] <= 5.0)
# mm.addConstr(-1./5.*xx[1] + xx[2] <= 2.0)
# mm.update()

# bc = BranchAndCut(initial_model=mm)
# bc.solve()
