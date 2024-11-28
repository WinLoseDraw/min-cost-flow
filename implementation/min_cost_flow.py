import json

import numpy as np

# Data class that contains the specifications to a min cost flow problem instance
class MinCostFlow:
    def __init__(self, nodes: int, demands: list[int], edges: list[tuple[int, int]], costs: list[int],
                 lower_capacities: list[int], upper_capacities: list[int]):
        self.n = nodes
        self.demands = np.array(demands, dtype=int)
        self.m = len(edges)
        self.edges = edges
        self.costs = np.array(costs, dtype=int)
        self.lower_capacities = np.array(lower_capacities, dtype=int)
        self.upper_capacities = np.array(upper_capacities, dtype=int)
        self.C = np.max(np.abs(self.costs))
        self.U = max(np.max(np.abs(self.lower_capacities)), np.max(np.abs(self.upper_capacities)))
        self.alpha = 1 / np.log2(1000 * self.m * self.U)
        self.B = np.zeros((self.m, self.n), dtype=int) # Edge incidence matrix
        for edge_idx in range(self.m):
            u, v = edges[edge_idx]
            self.B[edge_idx, u] = 1
            self.B[edge_idx, v] = -1
        self.undirected_edge_index_map = {} # Map that stores the undirected mapping of edges to their indices
        for edge_idx in range(self.m):
            self.add_edge_to_undirected_map(self.edges[edge_idx], edge_idx)
        self.min_ratio_cycle_finder = None

    def display_instance(self):
        print(f"Instance with n = {self.n}, m = {self.m}")
        print(f"Demands: {self.demands}")
        print(f"Edges: {self.edges}")
        print(f"Costs: {self.costs}")
        print(f"Lower capacities: {self.lower_capacities}")
        print(f"Upper capacities: {self.upper_capacities}")
        print(f"Undirected edge index map: {self.undirected_edge_index_map}")
        self.display_edge_incidence_matrix_B()

    def display_edge_incidence_matrix_B(self):
        print(f"Edge incidence matrix: ")
        print(f"{'vertices:': >10} ", ' '.join([f"{i: >2}" for i in range(self.n)]))
        for e, row in enumerate(self.B):
            print(f"{e: <2} {str(self.edges[e]): <7}", row)

    # Calculate the potential function of the flow, phi as specified in the paper
    def find_phi(self, flow: np.ndarray, optimal_flow_cost: int) -> float:
        flow_cost = np.dot(self.costs, flow)
        first = 20 * self.m * np.log2(flow_cost - optimal_flow_cost)
        left = (self.upper_capacities - flow) ** (-self.alpha)
        right = (flow - self.lower_capacities) ** (-self.alpha)
        sigma = np.sum(left + right)
        return first + sigma

    # Calculate the lengths of the flow as specified in the paper
    def find_lengths(self, flow: np.ndarray) -> np.ndarray:
        exponent = -1 - self.alpha
        first = (self.upper_capacities - flow) ** exponent
        second = (flow - self.lower_capacities) ** exponent
        return first + second

    # Calculate the gradients, derivative of the potential function phi, as specified in the paper
    def find_gradients(self, flow: np.ndarray, optimal_flow_cost: int) -> np.ndarray:
        flow_cost = np.dot(self.costs, flow)
        first = 20 * self.m * ((flow_cost - optimal_flow_cost) ** (-1)) * self.costs
        exponent = -1 - self.alpha
        second = self.alpha * (self.upper_capacities - flow) ** exponent
        third = -self.alpha * (flow - self.lower_capacities) ** exponent
        return first + second + third

    # Add a vertex to the end of the graph
    def add_vertex(self) -> int:
        self.n += 1
        self.demands = np.append(self.demands, 0)
        self.B = np.pad(self.B, ((0, 0), (0, 1)))
        return self.n - 1

    # Add an edge from u to v with a given cost, lower capacity and upper capacity
    def add_edge(self, u: int, v: int, cost: int, lower_capacity: int, upper_capacity: int):
        self.edges.append((u, v))
        self.add_edge_to_undirected_map((u, v), len(self.edges) - 1)
        self.m += 1
        self.costs = np.append(self.costs, cost)
        self.C = max(self.C, cost)
        self.lower_capacities = np.append(self.lower_capacities, lower_capacity)
        self.upper_capacities = np.append(self.upper_capacities, upper_capacity)
        self.U = max(self.U, abs(lower_capacity), abs(upper_capacity))
        self.alpha = 1 / np.log2(1000 * self.m * self.U)
        self.B = np.pad(self.B, ((0, 1), (0, 0)))
        self.B[-1, u] = 1
        self.B[-1, v] = -1

    # Handles adding an edge to the map storing the undirected edge-index mapping
    def add_edge_to_undirected_map(self, edge, edge_index):
        u, v = edge
        if (u, v) not in self.undirected_edge_index_map:
            self.undirected_edge_index_map[(u, v)] = []
        if (v, u) not in self.undirected_edge_index_map:
            self.undirected_edge_index_map[(v, u)] = []
        self.undirected_edge_index_map[(u, v)].append(edge_index)
        self.undirected_edge_index_map[(v, u)].append(edge_index)

    # Construct a min cost flow instance from a given json file
    @staticmethod
    def from_json(path: str):
        with open(path, "r") as file:
            data = json.load(file)
            I = MinCostFlow(nodes=len(data["demands"]), demands=data["demands"],
                            edges=[(edge[0], edge[1]) for edge in data["edges"]], costs=data["costs"],
                            lower_capacities=data["lower_capacities"], upper_capacities=data["upper_capacities"])
            return I
