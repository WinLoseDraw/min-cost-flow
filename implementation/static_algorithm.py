import numpy as np

from implementation.intial_point import find_initial_feasible_flow
from implementation.max_flow import MaxFlow
from implementation.min_cost_flow import MinCostFlow
from implementation.min_ratio_cycles import find_min_ratio_cycle

# Solve the min cost flow problem given an optimal cost guess,
# using an implementation similar to the static method as mentioned in the paper
def min_cost_flow_with_optimal_cost(I_original: MinCostFlow,
                                    optimal_cost: int):
    I_original.optimal_flow_cost = optimal_cost
    print("Initial instance: ")
    I_original.display_instance()
    last_idx = I_original.m
    I, current_flow = find_initial_feasible_flow(I_original)
    print("Feasible instance: ")
    I.display_instance()
    # print(f"Initial feasible flow: {current_flow}")
    # threshold = (I.m * I.U) ** -10 # Threshold given in the paper
    threshold = 1e-5  # Taking a larger threshold to terminate faster
    iteration = 0
    current_phi = I.find_phi(current_flow, optimal_cost)
    while np.dot(I.costs, current_flow) - I.optimal_flow_cost >= threshold:
        iteration += 1
        # print("Iteration", iteration)
        # print("Current Φ(f) = ", current_phi)
        # print(f"Current flow = {current_flow}")
        min_ratio, min_ratio_cycle = find_min_ratio_cycle(I, current_flow, optimal_cost)
        # print(f"Min ratio = {min_ratio})")
        # print(f"Min ratio cycle = {min_ratio_cycle}")
        current_flow += min_ratio_cycle
        # print(f"New flow after augmenting = {current_flow}")
        current_phi = I.find_phi(current_flow, optimal_cost)
        if not current_phi < float('inf'):
            # print("Phi is too large")
            break
    final_cost = np.dot(I.costs[:last_idx], np.round(current_flow[:last_idx]))
    return final_cost, np.round(current_flow[:last_idx])

# Guesses the optimal flow cost using binary search and solves the min cost flow problem instance
def find_min_cost_flow(
        I: MinCostFlow):
    min_possible_cost = - I.C * I.U
    max_possible_cost = I.C * I.U
    found_cost = None
    found_flow = None
    l = min_possible_cost
    r = max_possible_cost
    while l < r:
        mid = (l + r) // 2
        found_cost, found_flow = min_cost_flow_with_optimal_cost(I, mid)
        if found_cost > mid:
            l = mid + 1
        else:
            r = mid
    return found_cost, found_flow

# Solve the max flow problem instance using the static algorithm
def find_max_flow(I: MaxFlow):
    I = I.to_min_cost_flow()
    min_cost, min_cost_flow = find_min_cost_flow(I)
    max_flow_value = -min_cost
    max_flow = min_cost_flow[:-1]
    return max_flow_value, max_flow
