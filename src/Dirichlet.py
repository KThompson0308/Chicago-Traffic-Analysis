# Code From https://dp.tdhopper.com/dirichlet-distribution/ and https://dp.tdhopper.com/hdp-sampling/

from numpy.random import choice

class DirichletProcessSample():
    def __init__(self, base_measure, alpha):
        self.base_measure = base_measure
        self.alpha = alpha

        self.cache = []
        self.weights = []
        self.total_stick_used = 0.
    
    def __call__(self):
        remaining = 1.0 - self.total_stick_used
        i = DirichletProcessSample.roll_die(self.weights + [remaining])
        if i is not None and i < len(self.weights):
            return self.cache[i]
        else:
            stick_piece = beta(1, self.alpha).rvs() * remaining
            self.total_stick_used += stick_piece
            self.weights.append(stick_piece)
            new_value = self.base_measure()
            self.cache.append(new_value)
            return new_value
        
    @staticmethod
    def roll_die(weights):
        if weights:
            return choice(range(len(weights)), p=weights)
        else:
            return None


class HierarchicalDirichletProcessSample(DirichletProcessSample):
    def __init__(self, base_measure, alpha1, alpha2):
        first_level_dp = DirichletProcessSample(base_measure, alpha1)
        self.second_level_dp = DirichletProcessSample(first_level_dp, alpha2)
    
    def __call__(self):
        return self.second_level_dp()