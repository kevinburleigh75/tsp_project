from collections import deque

import gurobipy as grb
import math
import re
import random
import time

random.seed(42)

from dataset import Dataset
from graph import Graph


def text_keys(text):
    ## http://stackoverflow.com/a/5967539/7002167
    def atoi(text):
        return int(text) if text.isdigit() else text

    return [atoi(ss) for ss in re.split('(\d+)', str(text))]


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

        self.queue           = []

        self.vars          = None
        self.best_cost     = None
        self.best_model    = None

        self.max_queue_len = 0


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

        start_time = time.time()

        model          = grb.Model('tsp')
        xx             = model.addVars(self.edges, lb=0.0, ub=1.0, vtype=grb.GRB.CONTINUOUS, name='xx', obj=self.cost_by_edge)
        degree_constrs = model.addConstrs((xx.sum(node,'*') + xx.sum('*',node) == 2.0 for node in self.nodes), 'degree')
        model.update()

        model.setParam('OutputFlag', False)

        use_min = True
        self.queue.append((float('inf'), grb.Model.copy(model)))

        while len(self.queue) != 0:
            if len(self.queue) > self.max_queue_len:
                self.max_queue_len = len(self.queue)

            min_item = min(self.queue)
            max_item = max(self.queue)

            # if use_min:
            #     print('popping from queue - min')
            #     item = min(self.queue)
            # else:
            #     print('popping from queue - max')
            #     item = max(self.queue)
            # use_min = not use_min

            # if self.best_cost is None:
            #     print('popping from queue - max')
            #     item = max(self.queue)
            # else:
            #     print('popping from queue - min')
            #     item = min(self.queue)

            # item = min(self.queue)
            # item = max(self.queue)

            # use_min = random.randint(1, 100) > 90
            # if use_min:
            #     print('popping from queue - min')
            #     item = min(self.queue)
            # else:
            #     print('popping from queue - max')
            #     item = max(self.queue)
            # use_min = not use_min

            # if self.best_cost is None:
            #     print('popping from queue - max')
            #     item = max(self.queue)
            # else:
            #     if use_min:
            #         print('popping from queue - min')
            #         item = min(self.queue)
            #     else:
            #         print('popping from queue - max')
            #         item = max(self.queue)
            #     use_min = not use_min

            if self.best_cost is None:
                print('popping from queue - max')
                item = max(self.queue)
            else:
                use_min = random.randint(1, 100) > 0
                if use_min:
                    print('popping from queue - min')
                    item = min(self.queue)
                else:
                    print('popping from queue - max')
                    item = max(self.queue)

            self.queue.remove(item)
            (_, model) = item

            last_soln = None
            cur_soln  = None

            while True:
                model.update()
                print('elapsed: {:+1.5e} optimizing: id: {} bc: {} ql: {} mq: {} qmin: {:+1.5e} qmax: {:+1.5e} nc: {}'.format(
                    time.time() - start_time,
                    id(model),
                    self.best_cost,
                    len(self.queue),
                    self.max_queue_len,
                    min_item[0],
                    max_item[0],
                    len(model.getConstrs())
                ))
                model.optimize()

                last_soln = cur_soln

                if self.solution_is_infeasible(model):
                    print('  infeasible')
                    cur_soln = None
                    break

                cur_soln = model.getAttr('X')

                # ss = ' '.join(['{:+1.3e}'.format(val) for val in cur_soln])
                # print('cur soln: {}'.format(ss))

                # if last_soln is not None:
                #     for idx,cur_val in enumerate(cur_soln):
                #         last_val = last_soln[idx]
                #         delta = abs(last_val - cur_val)
                #         ind1 = '*' if delta > 1e-3 else ' '
                #         ind2 = '-' if (last_val != 1.0 and abs(1.0-last_val) < 1e-5) or (last_val != 0.0 and abs(0.0-last_val) < 1e-5) or (cur_val != 1.0 and abs(1.0-cur_val) < 1e-5) or (cur_val != 0.0 and abs(0.0-cur_val) < 1e-5) else ' '
                #         print('idx: {:4d} last: {:+1.5e} cur: {:+1.5e} delta: {:+1.5e} {} {}'.format(idx, last_val, cur_val, delta, ind1, ind2))
                #     # ss = ' '.join(['{}'.format('x' if abs(cur_soln[idx] - last_soln[idx]) > 1e-3 else ' ') for idx in xrange(len(cur_soln))])
                #     # print('deltas: {}'.format(ss))

                if last_soln == cur_soln:
                    print('=== NO PROGRESS ===')
                    # print('last: {}'.format(last_soln))
                    # print('cur:  {}'.format(cur_soln))
                    # raise StandardError('last soln == cur soln')

                print('value / status: {:+1.5e} {}'.format(model.getAttr('ObjVal'), model.status))

                if not self.solution_can_become_new_best(model):
                    print('  cannot become best')
                    break

                if self.solution_is_tour(model):
                    print('  tour')
                    if self.solution_is_new_best(model):
                        print('  new best')
                        self.update_best(model)
                    break

                # skip_constraints = False
                # for (other_objval, other_model) in self.queue:
                #     if model.getAttr('ObjVal') > other_objval:
                #         skip_constraints = True
                #         break

                # if skip_constraints:
                #     print('  skipping constraints')
                #     self.queue.append((model.getAttr('ObjVal'), model))
                #     break

                print('adding constraints')
                constraints_were_added = False

                ## Global
                print('adding comb constraints')
                constraints_were_added = self.add_comb_constraints(model)                | constraints_were_added
                print('adding integral subtour constraints')
                constraints_were_added = self.add_integral_subtour_constraints(model)    | constraints_were_added
                print('adding non-integral subtour constraints')
                constraints_were_added = self.add_nonintegral_subtour_constraints(model) | constraints_were_added

                ## Local
                print('adding gomory constraints')
                constraints_were_added = self.add_gomory_constraints(model)              | constraints_were_added
                print('adding objective constraints')
                constraints_were_added = self.add_objective_constraints(model)           | constraints_were_added

                # if not constraints_were_added:
                #     print('adding gomory constraints')
                #     constraints_were_added = self.add_gomory_constraints(model) | constraints_were_added

                if constraints_were_added:
                    print('constraints were added')
                    continue
                else:
                    print('no constraints were added')

                if not self.solution_is_integral(model):
                    print('adding branches')
                    models = self.create_branch_models(model)
                    if len(models) == 0:
                        raise StandardError('no branch models could be found')
                    for branch_model in models:
                        self.queue.append((model.getAttr('ObjVal'), branch_model))
                    break
                else:
                    raise StandardError('integral non-tour solution was left unconstrained')


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


    def add_objective_constraints(self, model):
        if self.solution_is_infeasible(model):
            return False

        obj_val = model.getAttr('ObjVal')
        ceil_obj_val  = math.ceil(obj_val)
        floor_obj_val = math.floor(obj_val)

        if (abs(obj_val - floor_obj_val) < 1e-6) or (abs(obj_val - ceil_obj_val) < 1e-6):
            return False

        print('  adding objective constraint ({},{},{:+1.5e})'.format(obj_val, ceil_obj_val, ceil_obj_val - obj_val))

        # ceil_obj_val = max(ceil_obj_val, 108000)

        ##
        ## NOTE: This is a local cut (valid only for the current model given
        ##       its current optimum) and as such should NOT be applied to
        ##       all models in the queue.
        ##

        target_constr_name = 'objective-roundup'

        target_constr = None
        for constr in model.getConstrs():
            if constr.getAttr('ConstrName') == target_constr_name:
                target_constr = constr
                break

        if target_constr is not None:
            target_constr.setAttr('RHS', ceil_obj_val)
        else:
            mvars = model.getVars()

            var_idxs = sorted([self.idx_by_edge[edge] for edge in self.edges])
            cvars = [mvars[idx].getAttr('Obj')*mvars[idx] for idx in var_idxs]

            model.addConstr(grb.quicksum(cvars) >= ceil_obj_val, 'objective-roundup')


        print('  objective constraints added')
        return True


    def add_integral_subtour_constraints(self, model):
        if not self.solution_is_integral(model):
            return False

        graph, xx = self.convert_model(model)
        connected_component_nodes, connected_component_edges = graph.binary_connected_components(solution=xx)

        if len(connected_component_nodes) == 1:
            return False

        models = [mm for (_, mm) in self.queue]
        models.append(model)

        for model in models:
            mvars = model.getVars()

            for cc in connected_component_nodes:
                cut_edges = graph.get_cut_edges(nodes=cc)
                var_idxs = sorted([self.idx_by_edge[edge] for edge in cut_edges])
                cvars = [mvars[idx] for idx in var_idxs]

                model.addConstr(grb.quicksum(cvars) >= 2.0, 'subtour-integral')

        print('  integral subtour constraints added')

        return True

    def add_nonintegral_subtour_constraints(self, model):
        if self.solution_is_integral(model):
            return False

        graph, xx = self.convert_model(model)


        modified_graph, xx = self.convert_model(model)
        all_cuts = []
        iter_num = 0
        while iter_num < 100:
            iter_num +=  1

            cur_cuts = modified_graph.find_min_cut()
            all_cuts.extend(cur_cuts)

            if cur_cuts[0][0] > 0.0:
                break

            new_nodes   = set(modified_graph.nodes) - set(cur_cuts[0][1])
            new_edges   = set()
            new_weights = {}
            for node in new_nodes:
                for edge in modified_graph.edges_by_node[node]:
                    if (edge[0] in new_nodes) and (edge[1] in new_nodes):
                        new_edges.add(edge)
                        new_weights[edge] = modified_graph.weight_by_edge[edge]

            modified_graph = Graph(nodes=new_nodes, edges=new_edges, weight_by_edge=new_weights)

        all_cuts = sorted(all_cuts, key=lambda x: x[0])

        # for idx,cut in enumerate(all_cuts):
        #     if cut[0] < 2.0 - 1e-8:
        #         print('cut {}: {:+1.5e} {}'.format(idx,cut[0],cut[1]))

        if all_cuts[0][0] >= 2.0 - 1e-8:
            return False

        models = [mm for (_, mm) in self.queue]
        models.append(model)

        for model in models:
            mvars = model.getVars()

            for (cut_value, cut_nodes) in all_cuts:
                if cut_value >= 2.0 - 1e-8:
                    break

                cut_edges = graph.get_cut_edges(nodes=cut_nodes)

                var_idxs = sorted([self.idx_by_edge[edge] for edge in cut_edges])
                cvars = [mvars[idx] for idx in var_idxs]

                model.addConstr(grb.quicksum(cvars) >= 2.0, 'subtour-nonintegral')


        print('  nonintegral subtour constraints added')

        return True


    def add_comb_constraints(self, model):
        graph, xx = self.convert_model(model)

        ##
        ## Find the connected components of the G12 graph, which
        ## is G with the edges with decision values 1.0,0.0 removed.
        ##

        g12_edges = [edge for edge in graph.edges if xx[graph.idx_by_edge[edge]] not in [1.0,0.0]]
        weight_by_edge = {edge: 0.0 for edge in g12_edges}
        g12 = Graph(nodes=graph.nodes, edges=g12_edges, weight_by_edge=weight_by_edge)
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

                    # print('comb - multiply intersected nodes: {}'.format(multiply_intersected_nodes))
                    for node in multiply_intersected_nodes:
                        cc_nodes.append(node)
                    cc_cut_edges = graph.get_cut_edges(nodes=cc_nodes)
                    one_edges = [edge for edge in cc_cut_edges if xx[graph.idx_by_edge[edge]] == 1.0]

                cc_nodes  = sorted(set(cc_nodes), key=text_keys)
                one_edges = sorted(set(one_edges), key=tuple_keys)

                if len(one_edges) == 1:
                    # print('comb - subtour')

                    handle_cut_edges = sorted(set(graph.get_cut_edges(nodes=cc_nodes)), key=tuple_keys)
                    handle_var_idxs  = sorted([self.idx_by_edge[edge] for edge in handle_cut_edges])

                    models = [mm for (_, mm) in self.queue]
                    models.append(model)

                    for midx,model in enumerate(models):

                        mvars = model.getVars()

                        handle_vars = [mvars[idx] for idx in handle_var_idxs]

                        model.addConstr(grb.quicksum(handle_vars) >= 2.0, name='comb-subtour')

                    constraints_were_added = True

                else:
                    # print('comb - blossom H={} T={}'.format(cc_nodes,one_edges))

                    handle_cut_edges = sorted(set(graph.get_cut_edges(nodes=cc_nodes)), key=tuple_keys)
                    tooth_cut_edges  = [graph.get_cut_edges(nodes=tooth_nodes) for tooth_nodes in one_edges]
                    tooth_cut_edges  = sorted([item for sublist in tooth_cut_edges for item in sublist], key=tuple_keys)

                    handle_var_idxs = sorted([self.idx_by_edge[edge] for edge in handle_cut_edges])
                    tooth_var_idxs  = sorted([self.idx_by_edge[edge] for edge in tooth_cut_edges])

                    models = [mm for (_, mm) in self.queue]
                    models.append(model)

                    for midx,model in enumerate(models):

                        mvars = model.getVars()

                        handle_vars = [mvars[idx] for idx in handle_var_idxs]
                        tooth_vars  = [mvars[idx] for idx in tooth_var_idxs]

                        expr = grb.quicksum(handle_vars) + grb.quicksum(tooth_vars) >= 3*len(one_edges) + 1

                        model.addConstr(expr, name='comb')

                    constraints_were_added = True

        if constraints_were_added:
            print('  comb constraints added')

        return constraints_were_added


    def add_gomory_constraints(self, model):
        if self.solution_is_integral(model):
            return False

        # print('starting Gomory cuts {}'.format(model.getAttr('ObjVal')))

        constraints_were_added = False

        target_constrs = set()
        for target_var_idx,target_var in enumerate(model.getVars()):
            if target_var.getAttr('VBasis') != 0:
                continue
            if target_var.getAttr('X') == int(target_var.getAttr('X')):
                continue
            col = model.getCol(target_var)
            for constr_idx in xrange(col.size()):
                constr = col.getConstr(constr_idx)
                if constr in target_constrs:
                    continue
                if constr.getAttr('Slack') != 0.0:
                    continue
                if constr.getAttr('RHS') != int(constr.getAttr('RHS')):
                    continue
                if constr.getAttr('RHS') == 0.0:
                    continue

                row = model.getRow(constr)
                # print('constr = {}'.format(constr))
                # print('row    = {}'.format(row))

                all_coeffs_are_integral = True
                for var_idx in xrange(row.size()):
                    coeff = row.getCoeff(var_idx)
                    if coeff != int(coeff):
                        all_coeffs_are_integral = False
                        break
                if not all_coeffs_are_integral:
                    continue

                target_constrs.add(constr)

        for constr in target_constrs:
            # if constraints_were_added:
            #     break

            rhs = constr.getAttr('RHS')
            row = model.getRow(constr)

            old_coeff_by_var = {var: 0.0 for var in model.getVars()}
            for var_idx in xrange(row.size()):
                var   = row.getVar(var_idx)
                coeff = row.getCoeff(var_idx)
                old_coeff_by_var[var] = coeff

            divisor = 0.0
            for var,coeff in old_coeff_by_var.items():
                if (coeff != 0.0) and (rhs/coeff != int(rhs/coeff)):
                    if abs(coeff) >= divisor:
                        divisor = abs(coeff)
            if divisor == 0.0:
                continue
                # raise StandardError('divisor is zero!')

            adj_coeff_by_var = {var: math.floor(coeff/divisor) for var,coeff in old_coeff_by_var.items()}

            if rhs/divisor == int(rhs/divisor):
                raise StandardError('integral RHS after dividing!')

            new_rhs = math.floor(rhs/divisor) - 1e-3
            new_coeff_by_var = {var: (old_coeff_by_var[var] - adj_coeff_by_var[var])/new_rhs for var in old_coeff_by_var}
            new_rhs = 1.0

            all_coeffs_are_zero = True
            for var,coeff in new_coeff_by_var.items():
                if coeff != 0.0:
                    all_coeffs_are_zero = False
                    break
            if all_coeffs_are_zero:
                continue

            lhs = 0.0
            for var,coeff in new_coeff_by_var.items():
                lhs += coeff*var.getAttr('X')

            if lhs >= new_rhs:
                continue

            # print('constraint is violated: {:+1.5e} !>= {:+1.5e} delta: {:+1.5e}'.format(lhs, new_rhs, lhs-new_rhs))
            # print('constraint: {:+1.5e} >= {:+1.5e} ({:+1.5e})'.format(lhs, new_rhs, lhs-new_rhs))

            ##
            ## Check for duplicate constraint.
            ##

            is_duplicate = False
            for constr in model.getConstrs():
                if is_duplicate:
                    break

                rhs_match = abs(constr.getAttr('RHS') - new_rhs) < 1e-3

                lhs_match = True
                row = model.getRow(constr)
                for var_idx in xrange(row.size()):
                    row_coeff = row.getCoeff(var_idx)
                    row_var   = row.getVar(var_idx)

                    if abs(row_coeff - new_coeff_by_var[row_var]) > 1e-3:
                        lhs_match = False
                        break

                is_duplicate = lhs_match and rhs_match

            expr_coeffs = []
            expr_vars   = []
            for var,coeff in new_coeff_by_var.items():
                if coeff != 0.0:
                    expr_coeffs.append(coeff)
                    expr_vars.append(var)

            expr = grb.LinExpr(expr_coeffs, expr_vars) >= new_rhs
            # print('  expr = {}'.format(expr))
            if is_duplicate:
                # print('  duplicate constraint found')
                continue

            model.addConstr(expr, name='gomory')

            constraints_were_added = True

        if constraints_were_added:
            print('  Gomory constraints added')

        return constraints_were_added

    def coeffs_to_expr(self, coeff_by_var, model):
        expr_coeffs = []
        expr_vars   = []
        for var in model.getVars():
            coeff = coeff_by_var[var]
            if coeff != 0.0:
                expr_coeffs.append(coeff)
                expr_vars.append(var)
        expr = grb.LinExpr(expr_coeffs, expr_vars)
        return expr




    def solution_is_integral(self, model):
        if grb.GRB.OPTIMAL != model.status:
            return False

        for mvar in model.getVars():
            val = mvar.getAttr('X')
            if val != int(val):
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

        best_idx = None
        best_var = None
        best_val = None

        for idx,mvar in enumerate(model.getVars()):
            val = mvar.getAttr('X')
            if abs(val - int(val)) != 0.0:
                if (best_val is None) or (abs(val - 0.5) < best_val):
                    best_val = abs(val - 0.5)
                    best_var = mvar
                    best_idx = idx

        print('(best_idx,best_val) = ({},{})'.format(best_idx,best_val))

        model1 = grb.Model.copy(model)
        m1var  = model1.getVarByName(best_var.getAttr('VarName'))
        model1.addConstr(m1var == 0.0)
        model1.update()

        model2 = grb.Model.copy(model)
        m2var  = model2.getVarByName(best_var.getAttr('VarName'))
        model2.addConstr(m2var == 1.0)
        model2.update()

        return (model1, model2)


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

    enable_profiler = False

    if enable_profiler:
        import cProfile, pstats, StringIO
        prof = cProfile.Profile()
        prof.enable()

    # nodes = range(6)
    # edges = []
    # for idx1 in xrange(len(nodes)-1):
    #     for idx2 in xrange(idx1+1,len(nodes)):
    #         edges.append((nodes[idx1],nodes[idx2]))

    # distance_by_edge = {}
    # for edge in edges:
    #     distance_by_edge[edge] = 1.0

    # distance_by_edge[(0,3)] = 2.0
    # distance_by_edge[(0,4)] = 2.0
    # distance_by_edge[(0,5)] = 2.0
    # distance_by_edge[(1,3)] = 2.0
    # distance_by_edge[(1,4)] = 2.0
    # distance_by_edge[(1,5)] = 2.0
    # distance_by_edge[(2,3)] = 2.0
    # distance_by_edge[(2,4)] = 2.0
    # distance_by_edge[(2,5)] = 2.0

    start_time = time.time()
    bc = TspBranchAndCut(nodes=nodes, edges=edges, cost_by_edge=distance_by_edge)
    bc.solve()
    end_time = time.time()
    print('BEST COST: {}'.format(bc.best_cost))
    print('elapsed: {:+1.5e}'.format(end_time - start_time))


    if enable_profiler:
        prof.disable()
        ss = StringIO.StringIO()
        ps = pstats.Stats(prof, stream=ss).sort_stats('cumulative')
        ps.print_stats()
        print ss.getvalue()
