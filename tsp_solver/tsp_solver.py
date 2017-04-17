from collections import deque, namedtuple

import gurobipy as grb
import logging
import math
import numpy as np
import re
import random
import time

from array_utils import FloatArrayFormatter, BooleanArrayFormatter
from dataset import Dataset
from graph import Graph

random.seed(42)


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

        self.queue = []

        self.best_cost  = None
        self.best_model = None

        self.max_model_pool_size = 0

        self.solve_start_time = None

        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.debug('NODES: {}'.format(' '.join(self.nodes)))
        self.logger.debug('EDGES: {}'.format(' '.join(['({} {})'.format(node1,node2) for node1,node2 in self.edges])))
        self.logger.debug('COSTS: {}'.format(' '.join(['{}'.format(self.cost_by_edge[edge]) for edge in self.edges])))


    def solve(self):
        self.solve_start_time = time.time()

        initial_model = self.create_initial_model()
        self.add_model_to_pool(model=initial_model, obj_lb=-float('inf'))

        while not self.model_pool_is_empty():
            model = self.remove_next_model_from_pool()

            for obj_lb,new_model in self.process_model(model):
                self.add_model_to_pool(model=new_model, obj_lb=obj_lb)

        self.solve_end_time = time.time()

        self.logger.info('BEST COST: {}'.format(self.best_cost))
        self.logger.info('elapsed: {:+1.5e}'.format(self.solve_end_time - self.solve_start_time))


    def create_initial_model(self):
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

        return model


    def process_model(self, model):
        last_soln = cur_soln = None

        new_model_info = []

        while True:
            model.update()
            model.optimize()
            self.record_stats(model)

            last_soln = cur_soln

            if self.solution_is_infeasible(model):
                cur_soln = None
                break

            cur_soln = model.getAttr('X')

            if cur_soln == last_soln:
                self.logger.warn('=== NO PROGRESS WAS MADE BY NEWLY ADDED CUTS ===')

            # num_nonbinary = len([val for val in cur_soln if (val != 0) and (val != 1)])
            # num_notclose  = len([val for val in cur_soln if (abs(val - 0) > 1e-6) and (abs(val - 1) > 1e-6)])
            # print('num nonbinary/nonclose: {} / {}'.format(num_nonbinary, num_notclose))

            # print('value / status: {:+1.5e} {}'.format(model.getAttr('ObjVal'), model.status))

            if not self.solution_can_become_new_best(model):
                break

            if self.solution_is_tour(model):
                if self.solution_is_new_best(model):
                    self.update_best(model)
                break

            if self.add_cuts_to_model(model):
                continue

            if self.solution_is_integral(model):
                raise StandardError('integral non-tour solution was left unimproved')

            branch_models = self.create_branch_models(model)
            if len(branch_models) == 0:
                raise StandardError('no branch models could be found')

            for branch_model in branch_models:
                new_model_info.append( (model.getAttr('ObjVal'), branch_model) )

            break

        return new_model_info


    def add_cuts_to_model(self, model):

        stop_on_first = False

        constraints_were_added = False

        if stop_on_first:
            constraints_were_added = constraints_were_added | self.add_objective_constraints(model)
            constraints_were_added = constraints_were_added | self.add_comb_constraints(model)
            constraints_were_added = constraints_were_added | self.add_integral_subtour_constraints(model)
            constraints_were_added = constraints_were_added | self.add_nonintegral_subtour_constraints(model)
            # constraints_were_added = constraints_were_added | self.add_gomory_constraints(model)
        else:
            new_constraints = self.add_comb_constraints(model)
            constraints_were_added = constraints_were_added | new_constraints

            new_constraints = self.add_integral_subtour_constraints(model)
            constraints_were_added = constraints_were_added | new_constraints

            new_constraints = self.add_nonintegral_subtour_constraints(model)
            constraints_were_added = constraints_were_added | new_constraints

            # new_constraints = self.add_gomory_constraints(model)
            # constraints_were_added = constraints_were_added | new_constraints

            new_constraints = self.add_objective_constraints(model)
            constraints_were_added = constraints_were_added | new_constraints

        return constraints_were_added


    def create_branch_models(self, model):
        if grb.GRB.OPTIMAL != model.status:
            raise StandardError('model was not solved to optimality')

        xx = model.getAttr('X')
        self.logger.debug('BRANCH SOLUTION: {}'.format(self.encode_solution(xx)))

        best_var  = None

        if True:
            ##
            ## Branch on the most costly nonintegral edge.
            ##

            best_idx  = None
            best_cost = None

            for idx,mvar in enumerate(model.getVars()):
                val = mvar.getAttr('X')
                if abs(val - int(val)) != 0.0:
                    edge = self.edge_by_idx[idx]
                    cost = self.cost_by_edge[edge]

                    if (best_cost is None) or (cost > best_cost):
                        best_cost = cost
                        best_var  = mvar
                        best_idx  = idx
        elif False:
            ##
            ## Branch on most costly edge of node with most costly non-zero edges.
            ##

            best_node = None
            best_edge = None
            best_cost = None

            cost_by_node  = {node: 0.0   for node in self.nodes}
            edges_by_node = {node: set() for node in self.nodes}

            for edge in self.edges:
                edge_idx = self.idx_by_edge[edge]
                if (xx[edge_idx] != 0.0) and (xx[edge_idx] != 1.0):
                    edge_cost = self.cost_by_edge[edge]
                    edges_by_node[edge[0]].add(edge)
                    edges_by_node[edge[1]].add(edge)
                    cost_by_node[edge[0]] += edge_cost
                    cost_by_node[edge[1]] += edge_cost

            best_node = max(cost_by_node, key=lambda node: cost_by_node[node])
            best_edge = max(edges_by_node[best_node], key=lambda edge: self.cost_by_edge[edge])

            best_var = model.getVars()[self.idx_by_edge[best_edge]]

        if best_var is None:
            raise StandardError('could not create branches on solution')
        # print('(best_idx,best_cost) = ({},{})'.format(best_idx,best_cost))

        model1 = grb.Model.copy(model)
        m1var  = model1.getVarByName(best_var.getAttr('VarName'))
        model1.addConstr(m1var == 0.0)
        model1.update()

        model2 = grb.Model.copy(model)
        m2var  = model2.getVarByName(best_var.getAttr('VarName'))
        model2.addConstr(m2var == 1.0)
        model2.update()

        return (model1, model2)


    def add_model_to_pool(self, model, obj_lb):
        self.queue.append((obj_lb, model))


    def remove_next_model_from_pool(self):
        if self.best_cost is None:
            self.logger.info('no best - popping from queue - max')
            item = max(self.queue)
        else:
            use_min = random.randint(1, 100) < 30
            if use_min:
                self.logger.info('popping from queue - min')
                item = min(self.queue)
            else:
                self.logger.info('popping from queue - max')
                item = max(self.queue)

        self.queue.remove(item)
        (_, model) = item

        if self.model_pool_size() > self.max_model_pool_size:
            self.max_model_pool_size = self.model_pool_size()

        return model


    def model_pool_size(self):
        return len(self.queue)


    def model_pool_is_empty(self):
        return self.model_pool_size() == 0


    def get_model_pool_obj_bounds(self):
        if self.model_pool_is_empty():
            min_obj_lb = -float('inf')
            max_obj_lb = -float('inf')
        else:
            min_obj_lb, _ = min(self.queue)
            max_obj_lb, _ = max(self.queue)

        return (min_obj_lb, max_obj_lb)


    def solution_is_infeasible(self, model):
        is_infeasible = grb.GRB.INFEASIBLE == model.status

        self.logger.debug('is_infeasible = {}'.format(is_infeasible))

        return is_infeasible


    def solution_can_become_new_best(self, model):
        if grb.GRB.OPTIMAL != model.status:
            can_become_new_best = False
        elif self.best_cost is None:
            can_become_new_best = True
        else:
            can_become_new_best = (model.getAttr('ObjVal') < self.best_cost)

        self.logger.debug('can become new best = {}'.format(can_become_new_best))

        return can_become_new_best


    def solution_is_tour(self, model):
        graph, xx = self.convert_model(model)
        is_tour = graph.is_tour(solution=xx)

        self.logger.debug('is tour = {}'.format(is_tour))

        return is_tour


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
            is_new_best = False
        elif self.best_cost is None:
            is_new_best = True
        else:
            is_new_best = model.getAttr('ObjVal') < self.best_cost

        return is_new_best


    def encode_solution(self, soln):
        data = []
        for xx in soln:
            if xx == 0.0:
                data.append('0')
            elif xx == 1.0:
                data.append('1')
            elif abs(xx - 0.0) < 1e-8:
                data.append('Z')
            elif abs(xx - 1.0) < 1e-8:
                data.append('N')
            elif abs(xx - 0.5) < 1e-8:
                data.append('5')
            elif (xx >= 0.0) and (xx <= 0.5):
                data.append('L')
            elif (xx >= 0.5) and (xx <= 1.0):
                data.append('H')
            else:
                data.append('E')
        return ''.join(data)


    def update_best(self, model):
        if grb.GRB.OPTIMAL != model.status:
            raise StandardError('model was not solved to optimality')

        new_best_cost = model.getAttr('ObjVal')
        new_best_xx   = model.getAttr('X')


        self.logger.debug('NEW BEST TOUR: {}'.format(self.encode_solution(new_best_xx)))
        self.logger.info('new best {} (old = {})'.format(new_best_cost, self.best_cost))

        self.best_cost  = new_best_cost
        self.best_model = model


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


    def convert_model(self, model):
        solution = model.getAttr('X')
        weight_by_edge = {edge: solution[idx] for idx,edge in enumerate(self.edges)}
        graph = Graph(nodes=self.nodes, edges=self.edges, weight_by_edge=weight_by_edge)

        return (graph, solution)


    def record_stats(self, model):
        min_obj_lb, max_obj_lb = self.get_model_pool_obj_bounds()

        self.logger.info('elapsed: {:+1.5e} optimized: id: {} bc: {} ql: {} mq: {} qmin: {:+1.5e} qmax: {:+1.5e} nc: {}'.format(
            time.time() - self.solve_start_time,
            id(model),
            self.best_cost,
            self.model_pool_size(),
            self.max_model_pool_size,
            min_obj_lb,
            max_obj_lb,
            len(model.getConstrs())
        ))


    def is_duplicate_constraint(self, model, vars, coeffs, rhs, sense):
        coeff_by_var = {var: coeffs[idx] for idx,var in enumerate(vars)}

        is_duplicate = False

        for constr in model.getConstrs():
            if constr.getAttr('Sense') != sense:
                continue
            if abs(constr.getAttr('RHS') - rhs) > 1e-6:
                continue

            row = model.getRow(constr)
            is_duplicate = True
            for var_idx in xrange(row.size()):
                row_var   = row.getVar(var_idx)
                row_coeff = row.getCoeff(var_idx)

                if row_var not in coeff_by_var:
                    is_duplicate = False
                    break
                elif abs(row_coeff - coeff_by_var[row_var]) > 1e-6:
                    is_duplicate = False
                    break

            if is_duplicate:
                break

        return False


    def add_objective_constraints(self, model):
        obj_val = model.getAttr('ObjVal')
        ceil_obj_val  = math.ceil(obj_val)
        floor_obj_val = math.floor(obj_val)

        if (abs(obj_val - floor_obj_val) < 1e-6) or (abs(obj_val - ceil_obj_val) < 1e-6):
            return False

        # print('  adding objective constraint ({},{},{:+1.5e})'.format(obj_val, ceil_obj_val, ceil_obj_val - obj_val))

        ##
        ## NOTE: This is a local cut (valid only for the current model given
        ##       its current optimum) and as such should NOT be applied to
        ##       all models in the queue.
        ##

        ##
        ## If there is already an object constraint on the current model,
        ## just adjust its RHS.  Otherwise add a new one.
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

        return True


    def add_integral_subtour_constraints(self, model):
        if not self.solution_is_integral(model):
            return False

        graph, xx = self.convert_model(model)
        connected_component_nodes, connected_component_edges = graph.binary_connected_components(solution=xx)

        if len(connected_component_nodes) == 1:
            return False

        ##
        ## These are global constraints, so we can add them
        ## to every model in the queue.
        ##

        models = [mm for (_, mm) in self.queue]
        models.append(model)

        constraints_were_added = False

        for cur_model in models:
            mvars = cur_model.getVars()

            for cc in connected_component_nodes:
                ##
                ## Convert the cut to variables and coefficients.
                ##

                cut_edges = graph.get_cut_edges(nodes=cc)
                var_idxs = sorted([self.idx_by_edge[edge] for edge in cut_edges])
                cvars  = [mvars[idx] for idx in var_idxs]
                coeffs = [1.0 for idx in var_idxs]

                ##
                ## Ensure that this constraint isn't a duplicate
                ##

                if self.is_duplicate_constraint(model=model, vars=cvars, coeffs=coeffs, sense='>', rhs=2.0):
                    continue

                ##
                ## Add the constraint to the current model.
                ##

                cur_model.addConstr(grb.quicksum(cvars) >= 2.0, 'subtour-integral')

                ## ...and set the flag if it's the model we started with.
                if cur_model is model:
                    constraints_were_added = True

        return constraints_were_added


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

        ##
        ## If none of the min-cuts were less than 2.0, the subtour
        ## constraints are being met, so we're done.
        ##

        if all_cuts[0][0] >= 2.0 - 1e-8:
            return False

        ##
        ## This helper will be handy in about 30 lines.
        ##

        def add_constr_to_model(model, var_idxs, check_for_dups=False):
            mvars  = model.getVars()
            cvars  = [mvars[idx] for idx in var_idxs]
            coeffs = [1.0        for idx in var_idxs]

            if check_for_dups:
                if self.is_duplicate_constraint(model=model, vars=cvars, coeffs=coeffs, sense='>', rhs=2.0):
                    return False

            model.addConstr(grb.quicksum(cvars) >= 2.0, 'subtour-nonintegral')
            return True

        ##
        ## Determine which, if any, of the new cuts are duplicates.
        ##

        constraints_were_added = False

        non_dup_cuts = []
        for (cut_value, cut_nodes) in all_cuts:
            if cut_value >= 2.0 - 1e-8:
                break

            cut_edges = graph.get_cut_edges(nodes=cut_nodes)
            var_idxs  = sorted([self.idx_by_edge[edge] for edge in cut_edges])

            if add_constr_to_model(model=model, var_idxs=var_idxs, check_for_dups=True):
                non_dup_cuts.append(var_idxs)
                constraints_were_added = True

        ##
        ## Because these are a global cuts, we can add them
        ## to every model in the queue.
        ##

        models = [mm for (_, mm) in self.queue]
        for cur_model in models:
            for var_idxs in non_dup_cuts:
                add_constr_to_model(model=cur_model, var_idxs=var_idxs)

        return constraints_were_added


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


    def get_cost_by_edge(self, edge):
        if edge in self.cost_by_edge:
            return self.cost_by_edge[edge]
        else:
            return self.cost_by_edge[(edge[1],edge[0])]


    def best_tour(self):
        if self.best_cost is None:
            raise StandardError('there is no best tour')

        graph, soln = self.convert_model(self.best_model)

        soln_edges = [graph.edge_by_idx[idx] for idx in xrange(len(soln)) if soln[idx] == 1.0]

        cur_edge = soln_edges[0]
        tour_edges = [ (cur_edge[0],cur_edge[1]), (cur_edge[1],cur_edge[0]) ]
        tour = [ (cur_edge[0], cur_edge[1], self.get_cost_by_edge(cur_edge)) ]

        while len(tour) != len(soln_edges):
            starting_node = cur_edge[1]

            edge_found = False
            for edge in soln_edges:
                if starting_node in edge:
                    # print('  edge: {}'.format(edge))
                    if edge in tour_edges:
                        continue

                    if edge[0] == starting_node:
                        tour_edge = (edge[0],edge[1])
                    else:
                        tour_edge = (edge[1],edge[0])

                    tour.append( (tour_edge[0], tour_edge[1], self.get_cost_by_edge(tour_edge)) )
                    tour_edges.append( (edge[0],edge[1]) )
                    tour_edges.append( (edge[1],edge[0]) )

                    cur_edge = tour_edge
                    edge_found = True
                    break

            if not edge_found:
                raise StandardError('best solution is apparently not a tour - doh!')

        return tour


if __name__ == '__main__':
    import logging
    import logging.config
    import sys

    logging.config.fileConfig('logging.conf')

    ##
    ## Read the dataset from stdin
    ##

    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as fd:
            dataset = Dataset(istream=fd)
    else:
        dataset = Dataset(istream=sys.stdin)

    edges            = dataset.edges
    nodes            = dataset.nodes
    distance_by_edge = dataset.distance_by_edge

    ##
    ## Solve the problem.
    ##

    enable_profiler = True

    if enable_profiler:
        import cProfile, pstats, StringIO
        prof = cProfile.Profile()
        prof.enable()

    tsp_bc = TspBranchAndCut(nodes=nodes, edges=edges, cost_by_edge=distance_by_edge)
    tsp_bc.solve()

    total_cost = 0.0
    for node1,node2,cost in tsp_bc.best_tour():
        print('{:<3s} {:<3s} {}'.format(node1, node2, cost))
        total_cost += cost
    print('The cost of the best tour is: {}'.format(total_cost))

    if enable_profiler:
        prof.disable()
        ss = StringIO.StringIO()
        ps = pstats.Stats(prof, stream=ss).sort_stats('cumulative')
        ps.print_stats()
        print ss.getvalue()
