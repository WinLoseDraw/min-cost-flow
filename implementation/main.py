from implementation import nx_algorithm
from implementation import static_algorithm
from implementation.max_flow import MaxFlow
from implementation.min_cost_flow import MinCostFlow

# Solve a min cost flow problem using the nx library algorithm
def min_cost_flow_nx(path: str):
    print(f"Running example {path}")
    I = MinCostFlow.from_json(path)
    min_cost, min_cost_flow = nx_algorithm.find_min_cost_flow(I)
    print(f"Min cost: {min_cost}, Min cost flow = {min_cost_flow}")
    return min_cost, min_cost_flow

# Solve a max flow problem using the nx library algorithm
def max_flow_nx(path: str):
    print(f"Running example {path}")
    I = MaxFlow.from_json(path)
    max_flow_value, max_flow = nx_algorithm.find_max_flow(I)
    print(f"Max flow value: {max_flow_value}, Max flow: {max_flow}")
    return max_flow_value, max_flow

# Solve a min cost flow problem using the static algorithm
def min_cost_flow_static(path: str):
    print(f"Running example {path}")
    I = MinCostFlow.from_json(path)
    min_cost, min_cost_flow = static_algorithm.find_min_cost_flow(I)
    print(f"Min cost: {min_cost}, Min cost flow = {min_cost_flow}")
    return min_cost, min_cost_flow

# Solve a max flow problem using the static algorithm
def max_flow_static(path: str):
    print(f"Running example {path}")
    I = MaxFlow.from_json(path)
    max_flow_value, max_flow = static_algorithm.find_max_flow(I)
    print(f"Max flow value: {max_flow_value}, Max flow: {max_flow}")
    return max_flow_value, max_flow

# Verify that the min cost flow solutions obtained by the static algorithm and nx library are identical
def min_cost_flow_test(path: str):
    print(f"Testing example {path}")
    I = MinCostFlow.from_json(path)
    min_cost_nx, min_cost_flow_nx = nx_algorithm.find_min_cost_flow(I)
    min_cost_static, min_cost_flow_static = static_algorithm.find_min_cost_flow(I)
    print(f"NX:\nMin cost: {min_cost_nx}, Min cost flow: {min_cost_flow_nx}")
    print(f"Static:\nMin cost: {min_cost_static}, Min cost flow: {min_cost_flow_static}")
    assert min_cost_nx == min_cost_static
    print("Test passed")

# Verify that the max flow solutions obtained by the static algorithm and nx library are identical
def max_flow_test(path: str):
    print(f"Testing example {path}")
    I = MaxFlow.from_json(path)
    max_flow_value_nx, max_flow_nx = nx_algorithm.find_max_flow(I)
    max_flow_value_static, max_flow_static = static_algorithm.find_max_flow(I)
    print(f"NX:\nMax flow value: {max_flow_value_nx}, Max flow: {max_flow_nx}")
    print(f"Static:\nMax flow value: {max_flow_value_static}, Max flow: {max_flow_static}")
    assert max_flow_value_nx == max_flow_value_static
    print("Test passed")


if __name__ == "__main__":
    min_cost_flow_test("../min_cost_flow_examples/example1.json")
    max_flow_test("../max_flow_examples/example1.json")
