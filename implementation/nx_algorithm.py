import networkx as nx
import numpy as np

from implementation.max_flow import MaxFlow
from implementation.min_cost_flow import MinCostFlow

# Solve the min cost flow problem using the networkx library's min_cost_flow method
def find_min_cost_flow(I: MinCostFlow):
    I.display_instance()
    if sum(I.lower_capacities) != 0:
        I = normalize_graph(I)
    G = nx.DiGraph()
    for idx in range(I.m):
        u, v = I.edges[idx]
        G.add_edge(u, v, capacity=I.upper_capacities[idx], weight=I.costs[idx])
    for node in range(I.n):
        G.nodes[node]['demand'] = -I.demands[node]
    flow_dict = nx.min_cost_flow(G)
    min_cost = nx.cost_of_flow(G, flow_dict)
    min_cost_flow = np.zeros(I.m, dtype=int)
    for u in flow_dict:
        for v in flow_dict[u]:
            edge_flow = flow_dict[u][v]
            for idx in range(I.m):
                if I.edges[idx] == (u, v):
                    min_cost_flow[idx] = edge_flow
    return min_cost, min_cost_flow

# Solve the max flow problem using the networkx library's method
def find_max_flow(I: MaxFlow):
    I = I.to_min_cost_flow()
    min_cost, min_cost_flow = find_min_cost_flow(I)
    max_flow_value = -min_cost
    max_flow = min_cost_flow[:-1]
    return max_flow_value, max_flow

# Adjust the graph to have zero lower capacities because the networkx library does not support lower capacities
def normalize_graph(I: MinCostFlow):
    adjusted_edges = []
    for idx in range(I.m):
        u, v = I.edges[idx]
        adjusted_capacity = I.upper_capacities[idx] - I.lower_capacities[idx]
        adjusted_edges.append((u, v, adjusted_capacity, I.costs[idx]))
        I.demands[u] += I.lower_capacities[idx]
        I.demands[v] -= I.lower_capacities[idx]
    I.edges = adjusted_edges
    return I
