from collections import deque
import gurobipy as grb
import math


from dataset import Dataset


class BranchAndCut(object):

    def __init__(self, initial_model):
        self.initial_model = grb.Model.copy(initial_model)
        self.queue         = deque()

        self.best_cost     = None
        self.best_model    = None


    def solve(self):
        self.queue.append(grb.Model.copy(self.initial_model))

        while len(self.queue) != 0:
            model = self.queue.popleft()

            model.update()
            model.optimize()

            print('='*40)
            print(self.model_to_str(model, indent=2))
            print('='*40)

            if self.solution_is_infeasible(model):
                print('    INFEASIBLE')
                continue
            if not self.solution_can_become_new_best(model):
                print('    CANNOT BECOME BEST')
                continue

            ## TODO: Add heuristic cuts here.

            if self.solution_is_integral(model):
                if self.solution_is_new_best(model):
                    self.update_best(model)
            else:
                models = self.find_gomory_cuts(model)
                self.queue.extend(models)


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


    def find_gomory_cuts(self, model):
        if grb.GRB.OPTIMAL != model.status:
            raise StandardError('model was not solved to optimality')

        for mvar in model.getVars():
            val = mvar.getAttr('X')
            if abs(val - int(val)) > 1e-8:
                model1 = grb.Model.copy(model)
                m1var  = model1.getVarByName(mvar.getAttr('VarName'))
                model1.addConstr(m1var <= math.floor(val))

                model2 = grb.Model.copy(model)
                m2var  = model2.getVarByName(mvar.getAttr('VarName'))
                model2.addConstr(m2var >= math.ceil(val))

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
    xx             = model.addVars(edges, lb=0.0, ub=1.0, vtype=grb.GRB.CONTINUOUS, name='xx', obj=distance_by_edge)
    degree_constrs = model.addConstrs((xx.sum(node,'*') + xx.sum('*',node) == 2.0 for node in nodes), 'degree')
    model.update()

    ##
    ## Solve the problem.
    ##

    bc = BranchAndCut(initial_model=model)
    bc.solve()

    import pdb; pdb.set_trace()
    print('hello')


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
