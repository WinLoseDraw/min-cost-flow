import json

from implementation.min_cost_flow import MinCostFlow

# Data class that contains the specifications to a max flow problem instance
class MaxFlow:
    def __init__(self, edges: list[tuple[int, int]], upper_capacities: list[int], source: int, sink: int,
                 lower_capacities: list[int] = None):
        self.edges = edges
        self.lower_capacities = lower_capacities
        self.upper_capacities = upper_capacities
        self.source = source
        self.sink = sink

    # Convert the max flow problem instance to its corresponding min cost flow problem instance
    def to_min_cost_flow(self):
        edges = self.edges + [(self.sink, self.source)] # Additional edge from sink to source
        num_nodes = max(max(edge[0], edge[1]) for edge in edges) + 1
        demands = [0] * num_nodes
        costs = [0] * len(edges)
        costs[-1] = -1 # All edges have zero cost except the new edge which has -1 cost
        if self.lower_capacities is None:
            lower_capacities = [0] * len(edges)
        else:
            lower_capacities = self.lower_capacities + [0]
        upper_capacities = self.upper_capacities + [sum(self.upper_capacities)] # The new edge has a capacity equal to the sum of all other upper capacities
        I = MinCostFlow(nodes=num_nodes, demands=demands, edges=edges, costs=costs, lower_capacities=lower_capacities,
                        upper_capacities=upper_capacities)
        return I

    # Construct a max flow instance from a given json file
    @staticmethod
    def from_json(path: str):
        with open(path, "r") as file:
            data = json.load(file)
            I = MaxFlow(edges=[(edge[0], edge[1]) for edge in data["edges"]],
                        lower_capacities=data["lower_capacities"] if "lower_capacities" in data else None,
                        upper_capacities=data["upper_capacities"], source=data["source"], sink=data["sink"])
            return I
