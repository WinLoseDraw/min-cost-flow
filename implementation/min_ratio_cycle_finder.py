import numpy as np

# Class that stores all the circulations in the graph,
# and finds the min ratio cycle in every iteration given the new gradients and lengths
class MinRatioCycleFinder:
    def __init__(self, circulations: list[np.ndarray]):
        self.circulations = circulations

    def find_min_ratio_cycle(self, gradients: np.ndarray, lengths: np.ndarray):
        min_ratio = float('inf')
        min_ratio_cycle = np.zeros(0, dtype=np.float64)
        for circulation in self.circulations:
            for direction in [1.0, -1.0]:
                delta = circulation * direction
                gd = np.dot(gradients, delta)
                ld = lengths * delta
                norm = np.sum(np.abs(ld))
                ratio = gd / norm
                if ratio < min_ratio:
                    min_ratio = ratio
                    min_ratio_cycle = circulation
        return min_ratio, min_ratio_cycle
