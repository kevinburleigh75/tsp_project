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

        # model.setParam('OutputFlag', False)
        model.setParam('LogToConsole', False)
        model.setParam('NumericalFocus', 3)

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
            self.logger.debug('CURRENT: {}'.format(self.encode_solution(cur_soln)))
            self.logger.debug('cur obj val = {} {:+1.6e}'.format(model.status, model.getAttr('ObjVal')))

            skip_constraints = False
            if cur_soln == last_soln:
                self.logger.warn('=== NO PROGRESS WAS MADE BY NEWLY ADDED CUTS ===')
                # skip_constraints = True

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

            if not skip_constraints:
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

        stop_on_first = True

        constraints_were_added = False

        if stop_on_first:
            if self.add_comb_constraints(model):
                constraints_were_added = True
            elif self.add_integral_subtour_constraints(model):
                constraints_were_added = True
            elif self.add_nonintegral_subtour_constraints(model):
                constraints_were_added = True
            elif self.add_objective_constraints(model):
                constraints_were_added = True
            elif self.add_gomory_constraints(model):
                constraints_were_added = True
        else:
            new_constraints = self.add_comb_constraints(model)
            constraints_were_added = constraints_were_added | new_constraints

            new_constraints = self.add_integral_subtour_constraints(model)
            constraints_were_added = constraints_were_added | new_constraints

            new_constraints = self.add_nonintegral_subtour_constraints(model)
            constraints_were_added = constraints_were_added | new_constraints

            new_constraints = self.add_gomory_constraints(model)
            constraints_were_added = constraints_were_added | new_constraints

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
                self.logger.warn('UNKNOWN VALUE: {:+1.16e}'.format(xx))
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

        return is_duplicate


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

        self.logger.debug('  added objective constraints')

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

        if constraints_were_added:
            self.logger.debug('  added integral subtour constraints')

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
            expr = grb.quicksum(cvars) >= 2.0
            # self.logger.debug('  NIS: expr = {}'.format(expr))
            model.addConstr(expr, 'subtour-nonintegral')
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

        if constraints_were_added:
            self.logger.debug('  added nonintegral subtour constraints')

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
                            if node not in cc_nodes:
                                external_node_intersection_count[node] += 1

                    multiply_intersected_nodes = [node for node,count in external_node_intersection_count.items() if count > 1]

                    if len(multiply_intersected_nodes) == 0:
                        break

                    self.logger.debug('    comb multiply intersected nodes: {}'.format(multiply_intersected_nodes))
                    for node in multiply_intersected_nodes:
                        cc_nodes.add(node)
                    cc_cut_edges = graph.get_cut_edges(nodes=cc_nodes)
                    one_edges = [edge for edge in cc_cut_edges if xx[graph.idx_by_edge[edge]] == 1.0]

                cc_nodes  = sorted(set(cc_nodes), key=text_keys)
                one_edges = sorted(set(one_edges), key=tuple_keys)

                if len(one_edges) == 1:
                    self.logger.debug('    comb subtour')

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
                    self.logger.debug('    comb blossom H={} T={}'.format(cc_nodes,one_edges))

                    handle_cut_edges = sorted(set(graph.get_cut_edges(nodes=cc_nodes)), key=tuple_keys)
                    tooth_cut_edges  = [graph.get_cut_edges(nodes=tooth_nodes) for tooth_nodes in one_edges]
                    tooth_cut_edges  = sorted([item for sublist in tooth_cut_edges for item in sublist], key=tuple_keys)

                    handle_var_idxs = sorted([self.idx_by_edge[edge] for edge in handle_cut_edges])
                    tooth_var_idxs  = sorted([self.idx_by_edge[edge] for edge in tooth_cut_edges])

                    models = [mm for (_, mm) in self.queue]
                    models.append(model)

                    # expr_coeffs = [1.0 for var in it.chain(handle_vars, tooth_vars)]
                    # expr_vars   = [var for var in it.chain(handle_vars, tooth_vars)]
                    expr_rhs    = 3*len(one_edges) + 1

                    for midx,model in enumerate(models):

                        mvars = model.getVars()

                        handle_vars = [mvars[idx] for idx in handle_var_idxs]
                        tooth_vars  = [mvars[idx] for idx in tooth_var_idxs]

                        expr = grb.quicksum(handle_vars) + grb.quicksum(tooth_vars) >= expr_rhs

                        model.addConstr(expr, name='comb')

                    constraints_were_added = True

        if constraints_were_added:
            self.logger.debug('  added comb constraints')

        return constraints_were_added

    def _model_to_matrices(self, model):
            ##
            ## A*x = b
            ## B*xB + N*xB = b
            ##
            ## B = [ BM BS ]
            ## xB = [xBM xBS].T
            ##
            ## BS*xBS + NS*xNS = bBS
            ## xBS = [ xB.T xS.T ].T
            ##

            basic_model_vars    = []
            nonbasic_model_vars = []
            for var in model.getVars():
                if var.getAttr('VBasis') == 0:
                    basic_model_vars.append(var)
                else:
                    nonbasic_model_vars.append(var)

            slack_constrs    = []
            for constr in model.getConstrs():
                if constr.getAttr('CBasis') == 0:
                    slack_constrs.append(constr)

            NBMV = len(basic_model_vars)          ## Number of Basic     Model Variables
            NNMV = len(nonbasic_model_vars)       ## Number of Non-basic Model Variables
            NSV  = len(slack_constrs)             ## Number of Slack Variables
            NBV  = NBMV + NSV                     ## Number of Basic Variables (model + slack)
            NMV  = NBMV + NNMV                    ## Number of Model Variables (basic + nonbasic)

            row_idx_by_constr = {constr: idx for idx,constr in enumerate(model.getConstrs())}
            constr_by_row_idx = {idx: constr for constr,idx in row_idx_by_constr.items()}

            B_col_idx_by_basic_model_var = {var: idx for idx,var in enumerate(basic_model_vars)}
            B_basic_model_var_by_col_idx = {idx: var for var,idx in B_col_idx_by_basic_model_var.items()}

            B_col_idx_by_slack_constr = {var: NBMV+idx for idx,var in enumerate(slack_constrs)}
            B_slack_constr_by_col_idx = {idx: var      for var,idx in B_col_idx_by_slack_constr.items()}

            N_col_idx_by_nonbasic_model_var = {var: idx for idx,var in enumerate(nonbasic_model_vars)}
            N_nonbasic_model_var_by_col_idx = {idx: var for var,idx in N_col_idx_by_nonbasic_model_var.items()}

            B_NBVxNBV = np.zeros((NBV,NBV))
            xB_NBVx1  = np.zeros((NBV,1))

            N_NBVxNNMV = np.zeros((NBV,NNMV))
            xN_NNMVx1  = np.zeros((NNMV,1))

            b_NBVx1 = np.zeros((NBV,1))

            for constr in model.getConstrs():
                row_idx = row_idx_by_constr[constr]
                row     = model.getRow(constr)

                b_NBVx1[row_idx,0] = constr.getAttr('RHS')

                for var_idx in xrange(row.size()):
                    var   = row.getVar(var_idx)
                    val   = var.getAttr('X')
                    coeff = row.getCoeff(var_idx)

                    if var.getAttr('VBasis') == 0:
                        col_idx = B_col_idx_by_basic_model_var[var]
                        B_NBVxNBV[row_idx,col_idx] = coeff
                        xB_NBVx1[col_idx,0]        = val
                    else:
                        col_idx = N_col_idx_by_nonbasic_model_var[var]
                        N_NBVxNNMV[row_idx,col_idx] = coeff
                        xN_NNMVx1[col_idx,0]        = val

                if constr.getAttr('CBasis') == 0:
                    col_idx = B_col_idx_by_slack_constr[constr]
                    B_NBVxNBV[row_idx,col_idx] = 1.0
                    xB_NBVx1[col_idx,0]        = constr.getAttr('Slack')

            ##
            ## Sanity check: B*xB + N*xN = b
            ##

            sanity_lhs = B_NBVxNBV.dot(xB_NBVx1) + N_NBVxNNMV.dot(xN_NNMVx1)
            sanity_rhs = b_NBVx1

            if not np.allclose(sanity_lhs, sanity_rhs):
                import pdb; pdb.set_trace()
                raise StandardError('sanity check failed')

            Matrices = namedtuple('Matrices',
                'B_NBVxNBV N_NBVxNNMV ' +
                'b_NBVx1 xB_NBVx1 xN_NNMVx1 ' +
                'constr_by_row_idx row_idx_by_constr ' +
                'B_basic_model_var_by_col_idx B_col_idx_by_basic_model_var ' +
                'B_slack_constr_by_col_idx B_col_idx_by_slack_constr ' +
                'N_nonbasic_model_var_by_col_idx N_col_idx_by_nonbasic_model_var ' +
                'NBMV NNMV NBV NSV NMV '
            )

            matrices = Matrices(
                constr_by_row_idx               = constr_by_row_idx,
                row_idx_by_constr               = row_idx_by_constr,
                B_NBVxNBV                       = B_NBVxNBV,
                N_NBVxNNMV                      = N_NBVxNNMV,
                b_NBVx1                         = b_NBVx1,
                xB_NBVx1                        = xB_NBVx1,
                xN_NNMVx1                       = xN_NNMVx1,
                B_basic_model_var_by_col_idx    = B_basic_model_var_by_col_idx,
                B_col_idx_by_basic_model_var    = B_col_idx_by_basic_model_var,
                B_slack_constr_by_col_idx       = B_slack_constr_by_col_idx,
                B_col_idx_by_slack_constr       = B_col_idx_by_slack_constr,
                N_nonbasic_model_var_by_col_idx = N_nonbasic_model_var_by_col_idx,
                N_col_idx_by_nonbasic_model_var = N_col_idx_by_nonbasic_model_var,
                NBMV = NBMV,
                NNMV = NNMV,
                NBV  = NBV,
                NSV  = NSV,
                NMV  = NMV,
            )

            return matrices


    def add_gomory_constraints(self, model):
        if self.solution_is_integral(model):
            return False

        self.logger.debug('GC: starting')

        mm = self._model_to_matrices(model)

        self.logger.debug('GC: matrices')

        Binv_NBVxNBV     = np.linalg.solve(mm.B_NBVxNBV, np.eye(mm.NBV))
        tableau_NBVxNNMV = np.dot(Binv_NBVxNBV, mm.N_NBVxNNMV)

        self.logger.debug('GC: tableau')

        constraints_were_added = False

        for constr in model.getConstrs():
            if constraints_were_added:
                break

            ##
            ## Skip constraints whose slack variable is in the basis,
            ## since it could be anything
            ##

            if constr.getAttr('CBasis') == 0:
                # print('  GC: constr has basic slack variable')
                continue

            row_idx = mm.row_idx_by_constr[constr]
            if row_idx >= mm.NBMV:
                self.logger.debug('  GC: row_idx {} >= NBMV {}'.format(row_idx, mm.NBMV))
                continue
            # print('  GC: constr name = {}'.format(mm.A_constr_by_row_idx[row_idx].getAttr('ConstrName')))

            ##
            ## Skip previously-generated Gomory cuts.
            ##

            if constr.getAttr('ConstrName') == 'gomory':
                self.logger.debug('  GC: skipping gomory constraint')
                continue

            ##
            ## Check if the basic variable is non-integral.
            ## If not, move on.
            ##

            def round(xx):
                return math.floor(xx + 0.5)

            basic_val = mm.xB_NBVx1[row_idx,0]
            self.logger.debug('  GC: basic_val = {:+1.16e}'.format(basic_val))
            if abs(basic_val - round(basic_val)) < 1e-6:
                self.logger.debug('  GC: integral-ish basic_val')
                continue

            Binv_b = np.asscalar(np.dot(Binv_NBVxNBV[[row_idx],:], mm.b_NBVx1))
            self.logger.debug('  GC: Binv_b = {:+1.16e}'.format(Binv_b))

            frac_basic  = basic_val - math.floor(basic_val)
            frac_Binv_b = Binv_b    - math.floor(Binv_b)

            self.logger.debug('  GC: frac_basic  = {:+1.16e}'.format(frac_basic))
            self.logger.debug('  GC: frac_Binv_b = {:+1.16e}'.format(frac_Binv_b))

            ##
            ## Sanity check.
            ##

            sanity_lhs = basic_val
            sanity_rhs = Binv_b
            self.logger.debug('  GC: initial sanity_rhs {:+1.5e}'.format(sanity_rhs))

            for col_idx in xrange(mm.NNMV):
                nonbasic_var   = mm.N_nonbasic_model_var_by_col_idx[col_idx]
                if nonbasic_var.getAttr('VBasis') == 0:
                    raise StandardError('variable should not be basic')
                nonbasic_val   = nonbasic_var.getAttr('X')
                nonbasic_coeff = tableau_NBVxNNMV[row_idx,col_idx]

                # if abs(nonbasic_coeff) < 1e-6:
                #     continue

                # self.logger.debug('    GC: coeff/value {:+1.6e} {:+1.6e}'.format(nonbasic_coeff, nonbasic_val))

                if (nonbasic_val != nonbasic_var.getAttr('LB')) and (nonbasic_val != nonbasic_var.getAttr('UB')):
                    raise StandardError('nonbasic variable should be at one of its bounds: {}'.format(nonbasic_var))

                sanity_rhs -= nonbasic_coeff * nonbasic_val

            self.logger.debug('  GC: sanity: lhs {:+1.16e} rhs {:+1.16e}'.format(sanity_lhs, sanity_rhs))
            if abs(sanity_rhs - sanity_lhs) > 1e-8:
                self.logger.debug('  GC: sanity failure')
                import pdb; pdb.set_trace()
                raise StandardError('sanity check failed')

            ##
            ## Create four sets of nonbasic model variables:
            ##   - JJ0: at lower bound, tableau coeff < 1 - frac_basic
            ##   - JJ1: at lower bound, tableau coeff > 1 - frac_basic
            ##   - KK0: at upper bound, tableau coeff < frac_basic
            ##   - KK1: at upper bound, tableau coeff > frac_basic
            ##

            JJ0 = []
            JJ1 = []
            KK0 = []
            KK1 = []
            for col_idx in xrange(mm.NNMV):
                nonbasic_var   = mm.N_nonbasic_model_var_by_col_idx[col_idx]
                nonbasic_val   = nonbasic_var.getAttr('X')
                nonbasic_coeff = tableau_NBVxNNMV[row_idx,col_idx]

                # if abs(nonbasic_coeff) < 1e-6:
                #     continue

                lb = nonbasic_var.getAttr('LB')
                ub = nonbasic_var.getAttr('UB')
                if lb != 0.0:
                    raise StandardError('incorrect lower bound: {:+1.6e}'.format(lb))
                if ub != 1.0:
                    raise StandardError('incorrect upper bound: {:+1.6e}'.format(ub))

                fi = nonbasic_coeff - math.floor(nonbasic_coeff)

                if nonbasic_val == lb:
                    if fi <= 1.0 - frac_basic:
                        JJ0.append( (nonbasic_var,nonbasic_coeff) )
                    else:
                        JJ1.append( (nonbasic_var,nonbasic_coeff) )
                elif nonbasic_val == ub:
                    if fi <= frac_basic:
                        KK0.append( (nonbasic_var,nonbasic_coeff) )
                    else:
                        KK1.append( (nonbasic_var,nonbasic_coeff) )
                else:
                    raise StandardError('nonbasic variable {:+1.6e} not at either bound'.format(nonbasic_var))

            if (len(JJ0) == 0) and (len(JJ1) == 0) and (len(KK0) == 0) and (len(KK1) == 0):
                self.logger.debug('  GC: no nontrivial coeffs in JJ0,JJ0,KK0,KK1')
                continue

            ##
            ## Create a mixed-integer Gomory cut, and ensure
            ## that it is really violated by the current solution.
            ##

            expr_coeffs = []
            expr_vars   = []
            expr_lhs    = 0.0
            expr_rhs    = 1.0

            # self.logger.debug('    GC: JJ0:')
            for var, coeff in JJ0:
                fi = coeff - math.floor(coeff)
                new_coeff = fi / (1.0 - frac_basic)
                expr_coeffs.append(new_coeff)
                expr_vars.append(var)

                expr_lhs += new_coeff * var.getAttr('X')
                ## lower bound is zero so nothing to add to rhs

                # self.logger.debug('      GC: {:+1.16e} {}'.format(new_coeff, var))

            # self.logger.debug('    GC: JJ1:')
            for var, coeff in JJ1:
                fi = coeff - math.floor(coeff)
                new_coeff = (1.0 - fi) / frac_basic
                expr_coeffs.append(new_coeff)
                expr_vars.append(var)

                expr_lhs += new_coeff * var.getAttr('X')
                ## lower bound is zero so nothing to add to rhs

                # self.logger.debug('      GC: {:+1.16e} {}'.format(new_coeff, var))

            # self.logger.debug('    GC: KK0:')
            for var, coeff in KK0:
                fi = coeff - math.floor(coeff)
                new_coeff = -fi / frac_basic
                expr_coeffs.append(new_coeff)
                expr_vars.append(var)

                expr_lhs += new_coeff * var.getAttr('X')
                expr_rhs += new_coeff

                # self.logger.debug('      GC: {:+1.16e} {}'.format(new_coeff, var))

            # self.logger.debug('    GC: KK1:')
            for var, coeff in KK1:
                fi = coeff - math.floor(coeff)
                new_coeff = -(1.0 - fi) / (1.0 - frac_basic)
                expr_coeffs.append(new_coeff)
                expr_vars.append(var)

                expr_lhs += new_coeff * var.getAttr('X')
                expr_rhs += new_coeff

                # self.logger.debug('      GC: {:+1.16e} {}'.format(new_coeff, var))

            nonzero_coeff_found = False
            for coeff in expr_coeffs:
                if abs(coeff) > 1e-6:
                    nonzero_coeff_found = True
                    break
            if not nonzero_coeff_found:
                self.logger.debug('  GC: no nonzero coefficients in final lhs')
                continue

            ##
            ## Create four sets of nonbasic model variables:
            ##   - JJp: at lower bound, tableau coeff > 0
            ##   - JJn: at lower bound, tableau coeff < 0
            ##   - KKp: at upper bound, tableau coeff > 0
            ##   - KKn: at upper bound, tableau coeff < 0
            ##

            # JJp = []
            # JJn = []
            # KKp = []
            # KKn = []
            # for col_idx in xrange(mm.NNMV):
            #     nonbasic_var   = mm.N_nonbasic_model_var_by_col_idx[col_idx]
            #     nonbasic_val   = nonbasic_var.getAttr('X')
            #     nonbasic_coeff = tableau_NBVxNNMV[row_idx,col_idx]

            #     if abs(nonbasic_coeff) < 1e-10:
            #         continue

            #     lb = nonbasic_var.getAttr('LB')
            #     ub = nonbasic_var.getAttr('UB')
            #     if lb != 0.0:
            #         raise StandardError('incorrect lower bound: {:+1.6e}'.format(lb))
            #     if ub != 1.0:
            #         raise StandardError('incorrect upper bound: {:+1.6e}'.format(ub))

            #     if nonbasic_val == lb:
            #         if nonbasic_coeff > 0.0:
            #             JJp.append( (nonbasic_var,nonbasic_coeff) )
            #         elif nonbasic_coeff < 0.0:
            #             JJn.append( (nonbasic_var,nonbasic_coeff) )
            #     elif nonbasic_val == ub:
            #         if nonbasic_coeff > 0.0:
            #             KKp.append( (nonbasic_var,nonbasic_coeff) )
            #         elif nonbasic_coeff < 0.0:
            #             KKn.append( (nonbasic_var,nonbasic_coeff) )
            #     else:
            #         raise StandardError('nonbasic variable {:+1.6e} not at either bound'.format(nonbasic_var))

            # if (len(JJp) == 0) and (len(JJn) == 0) and (len(KKp) == 0) and (len(KKn) == 0):
            #     self.logger.debug('  GC: non nontrivial coeffs in JJn,JJp,KKn,KKp')
            #     continue

            # ##
            # ## Create a mixed-integer Gomory cut, and ensure
            # ## that it is really violated by the current solution.
            # ##

            # Binv_b_quant = frac_Binv_b*(2.0*frac_basic - 1.0)/(frac_basic*(1.0 - frac_basic))
            # self.logger.debug('  GC: Binv_b_quant = {:+1.16e}'.format(Binv_b_quant))

            # expr_coeffs = []
            # expr_vars   = []
            # expr_lhs    = 0.0 #Binv_b_quant
            # expr_rhs    = 1.0 - 1e-6

            # self.logger.debug('    GC: JJp:')
            # for var, coeff in JJp:
            #     new_coeff = coeff / (1.0 - frac_basic)
            #     expr_coeffs.append(new_coeff)
            #     expr_vars.append(var)

            #     expr_lhs += new_coeff * var.getAttr('X')
            #     ## lower bound is zero so nothing to add to rhs

            #     self.logger.debug('      GC: {:+1.16e} {}'.format(new_coeff, var))

            # self.logger.debug('    GC: JJn:')
            # for var, coeff in JJn:
            #     new_coeff = -coeff / frac_basic
            #     expr_coeffs.append(new_coeff)
            #     expr_vars.append(var)

            #     expr_lhs += new_coeff * var.getAttr('X')
            #     ## lower bound is zero so nothing to add to rhs

            #     self.logger.debug('      GC: {:+1.16e} {}'.format(new_coeff, var))

            # self.logger.debug('    GC: KKp:')
            # for var, coeff in KKp:
            #     new_coeff = -coeff / frac_basic
            #     expr_coeffs.append(new_coeff)
            #     expr_vars.append(var)

            #     expr_lhs += new_coeff * var.getAttr('X')
            #     expr_rhs += new_coeff

            #     self.logger.debug('      GC: {:+1.16e} {}'.format(new_coeff, var))

            # self.logger.debug('    GC: KKn:')
            # for var, coeff in KKn:
            #     new_coeff = coeff / (1.0 - frac_basic)
            #     expr_coeffs.append(new_coeff)
            #     expr_vars.append(var)

            #     expr_lhs += new_coeff * var.getAttr('X')
            #     expr_rhs += new_coeff

            #     self.logger.debug('      GC: {:+1.16e} {}'.format(new_coeff, var))

            self.logger.debug('  GC: lhs {:+1.16e} rhs {:+1.16e}'.format(expr_lhs, expr_rhs))
            if expr_lhs >= expr_rhs:
                self.logger.debug('  GC: lhs >= rhs')
                continue
                # raise StandardError('  GC: cut is not violated: {:+1.16e} is not >= {:+1.16e}'.format(expr_lhs, expr_rhs))

            if self.is_duplicate_constraint(model=model, vars=expr_vars, coeffs=expr_coeffs, sense='>', rhs=expr_rhs):
                self.logger.debug('  GC: duplicate constraint')
                continue

            expr = grb.LinExpr(expr_coeffs, expr_vars) >= expr_rhs
            # print('GC: constraint: {}'.format(expr))

            model.addConstr(expr, name='gomory')

            self.logger.debug('  GC: added constraint')

            constraints_were_added = True

        return constraints_were_added

    # def add_gomory_constraints(self, model):
    #     if self.solution_is_integral(model):
    #         return False

    #     mm = self._model_to_matrices(model)

    #     A_NBVxNV = np.concatenate((mm.B_NBVxNBV, mm.N_NBVxNNMV), axis=1)

    #     for var in model.getVars():
    #         if var.getAttr('VBasis'):
    #             A_col_idx = mm.B_col_idx_by_basic_model_var[var]
    #         else:
    #             A_col_idx = mm.NBV + mm.N_col_idx_by_nonbasic_model_var[var]

    #         nonzero_coeff_row_idxs = np.where(A_NBVxNV[:,A_col_idx] != 0)
    #         if len(nonzero_coeff_row_idxs) < 2:
    #             continue



    #         mvar_val = mvar.getAttr('X')
    #         if abs(mvar_val - int(mvar_val)) < 1e-8:
    #             continue

    #         col = model.getCol(mvar)



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

    enable_profiler = False

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
