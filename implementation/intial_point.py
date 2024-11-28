import copy

import numpy as np

from implementation.min_cost_flow import MinCostFlow

# Modify the problem instance as stated in the paper and obtain an initial feasible flow to the min cost flow problem
def find_initial_feasible_flow(I_original: MinCostFlow):
    I = copy.deepcopy(I_original)
    v_star = I.add_vertex() # New vertex
    initial_flow = (I.lower_capacities + I.upper_capacities) / 2
    original_demands = I_original.demands
    new_demands = np.dot(I.B.transpose(), initial_flow)
    new_cost = 4 * I_original.m * I_original.U ** 2 # Cost of the new edges to be added
    for node in range(I_original.n):
        d = original_demands[node]
        d_bar = new_demands[node]
        if d_bar > d:
            I.add_edge(v_star, node, new_cost, 0, 2 * (d_bar - d))
            initial_flow = np.append(initial_flow, d_bar - d)
        elif d_bar < d:
            I.add_edge(node, v_star, new_cost, 0, 2 * (d - d_bar))
            initial_flow = np.append(initial_flow, d - d_bar)
    return I, initial_flow
