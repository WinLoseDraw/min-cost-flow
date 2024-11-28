from itertools import combinations

import networkx as nx
import numpy as np

from min_cost_flow import MinCostFlow
from min_ratio_cycle_finder import MinRatioCycleFinder

# Find the min ratio cycle for a given instance using the current flow and the optimal cost value
def find_min_ratio_cycle(I: MinCostFlow, flow: np.ndarray, optimal_flow_cost: int):
    gradients = I.find_gradients(flow, optimal_flow_cost)
    lengths = I.find_lengths(flow)
    if I.min_ratio_cycle_finder is None: # Find and store all the circulations upon instantiation
        cycles = find_all_cycles(I)
        # print(f"Found {len(cycles)} cycles")
        circulations = get_circulations(I, cycles)
        I.min_ratio_cycle_finder = MinRatioCycleFinder(circulations)
    min_ratio, min_ratio_cycle = I.min_ratio_cycle_finder.find_min_ratio_cycle(gradients, lengths)
    assert min_ratio_cycle is not None and min_ratio < float('inf'), "No min ratio cycle found"
    gd = gradients.dot(min_ratio_cycle)
    eta = -10 / gd  # TODO: Scale according to the paper
    return min_ratio, min_ratio_cycle * eta

# Convert all the cycles stored into circulations
def get_circulations(I: MinCostFlow, cycles: list[list[int]]) -> list[np.ndarray]:
    circulations = []
    for cycle in cycles:
        circulation = np.zeros(I.m, dtype=np.float64)
        for idx in range(len(cycle)):
            edge = cycle[idx]
            nex = cycle[(idx + 1) % len(cycle)]
            a, b = I.edges[edge]
            if I.edges[nex][0] == a or I.edges[nex][1] == a:
                circulation[edge] = I.B[edge, a]
            else:
                circulation[edge] = I.B[edge, b]
        circulations.append(circulation)
    return circulations

# Find all the cycles in the given graph instance
def find_all_cycles(I: MinCostFlow):
    G = nx.MultiGraph(I.edges)
    visited_cycles = set()
    cycles = []
    pair_cycles = []
    for cycle in nx.simple_cycles(G):
        cycle_edges = []
        pair_edges = set()
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            edges = I.undirected_edge_index_map[(a, b)]
            if len(edges) > 1:
                pair_edges.add(edges[0])
            cycle_edges.append(edges[0])
        if len(pair_edges) > 0:
            pair_cycles.append((cycle_edges, pair_edges))
        if len(cycle_edges) == 2 and cycle_edges[0] == cycle_edges[1]:
            continue
        check_if_visited(cycle_edges, visited_cycles, cycles)
    cycles.extend(handle_pair_cycles(I, visited_cycles, pair_cycles))
    return cycles

# Handle cycles formed that include parallel edges
def handle_pair_cycles(I: MinCostFlow, visited_cycles: set[list[int]], pair_cycles: list[tuple[list[int], set[int]]]):
    cycles = []
    for (cycle, pair_edges) in pair_cycles:
        for amount_to_replace in range(1, len(pair_edges) + 1):
            for edges_to_replace in combinations(pair_edges, amount_to_replace):
                G = nx.MultiGraph(resolve_edges(I, cycle))
                for new_cycle in nx.simple_cycles(G):
                    cycle_edges = []
                    for i in range(len(new_cycle)):
                        a = new_cycle[i]
                        b = new_cycle[(i + 1) % len(new_cycle)]
                        edges = I.undirected_edge_index_map[(a, b)]
                        edges = [e for e in edges if e not in edges_to_replace]
                        cycle_edges.append(edges[0])
                    if len(cycle_edges) == 2 and cycle_edges[0] == cycle_edges[1]:
                        continue
                    check_if_visited(cycle_edges, visited_cycles, cycles)
        for edge in pair_edges:
            a, b = I.edges[edge]
            check_if_visited(I.undirected_edge_index_map[(a, b)], visited_cycles, cycles)
    return cycles

# Resolve the edge indices into their corresponding edges
def resolve_edges(I: MinCostFlow, edge_indices: list[int]):
    return [I.edges[a] for a in edge_indices]

# Add the cycle to the list of cycles only if it is not visited
def check_if_visited(cycle_edges, visited_cycles: set, cycles: list[int]):
    sorted_cycle_edges = tuple(sorted(cycle_edges))
    if sorted_cycle_edges not in visited_cycles:
        visited_cycles.add(sorted_cycle_edges)
        cycles.append(cycle_edges)
