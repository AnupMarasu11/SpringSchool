import numpy as np
import cma
from ..data.simulation import Simulation, SimulationData, CoilConfig
from .base import BaseOptimizer

from functools import partial
import multiprocessing as mp

def evaluate_solution(x, optimizer, simulation):
    config = optimizer._apply_parameters(x)
    sim_data = simulation(config)
    cost = optimizer.cost_function(sim_data)
    return (x, cost)

class CMAESOptimizer(BaseOptimizer):
    """
    CMA-ES Optimizer for MRI Coil Configuration.
    Optimizes phase and amplitude of 8 dipoles to maximize cost function.
    """
    def __init__(self, cost_function, sigma=0.3, max_iter=100):
        super().__init__(cost_function)
        self.sigma = sigma
        self.max_iter = max_iter

    def _vector_to_config(self, x):
        """Convert flat 16-dim vector to CoilConfig"""
        amplitude = np.clip(x[:8], 0, 1)
        phase = np.mod(x[8:], 2 * np.pi)
        return CoilConfig(amplitude=amplitude, phase=phase)

    def _apply_parameters(self, x: np.ndarray) -> CoilConfig:
        amplitude = x[:8]
        phase = x[8:]
        return CoilConfig(phase=phase, amplitude=amplitude)
    '''
    def optimize(self, simulation):
        x0 = np.concatenate([
            np.random.uniform(0.0, 1.0, 8),
            np.random.uniform(0.0, 2 * np.pi, 8)
        ])

        es = cma.CMAEvolutionStrategy(x0, self.sigma, {
            'maxiter': self.max_iter,
            'bounds': [
                np.concatenate([np.zeros(8), np.zeros(8)]),
                np.concatenate([np.ones(8), 2 * np.pi * np.ones(8)])
            ]
        })

        best_cost = -np.inf if self.direction == "maximize" else np.inf
        best_config = None

        # Pre-bind self and simulation using partial
        eval_func = partial(evaluate_solution, optimizer=self, simulation=simulation)

        with mp.Pool() as pool:
            while not es.stop():
                solutions = es.ask()
                results = pool.map(eval_func, solutions)

                costs = []
                for x, cost in results:
                    if (self.direction == "maximize" and cost > best_cost) or \
                    (self.direction == "minimize" and cost < best_cost):
                        best_cost = cost
                        best_config = self._apply_parameters(x)

                    costs.append(-cost if self.direction == "maximize" else cost)

                es.tell(solutions, costs)
                es.disp()

        return best_config
    '''
    
    def optimize(self, simulation):
        x0 = np.concatenate([np.random.uniform(low=0.0, high=1.0, size=8), np.random.uniform(low=0.0, high=2 * np.pi, size=8)])

        '''
        def objective(x):
            config = self._vector_to_config(x)
            sim_data = simulation(config)
            cost = self.cost_function(sim_data)
            return -cost if self.direction == "maximize" else cost

        es = cma.CMAEvolutionStrategy(x0, self.sigma, {'maxiter': self.max_iter})
        es.optimize(objective)

        best_solution = es.result.xbest
        return self._vector_to_config(best_solution)
        '''
        es = cma.CMAEvolutionStrategy(x0, self.sigma, {
            'maxiter': self.max_iter,
            'bounds': [np.concatenate([np.zeros(8), np.zeros(8)]),  # lower bounds
                        np.concatenate([np.ones(8), 2*np.pi*np.ones(8)])]  # upper bounds
            })

        best_cost = -np.inf if self.direction == "maximize" else np.inf
        best_config = None

        while not es.stop():
            solutions = es.ask()
            costs = []

            for x in solutions:
                config = self._apply_parameters(x)
                simulation_data = simulation(config)
                cost = self.cost_function(simulation_data)

                if (self.direction == "maximize" and cost > best_cost) or (self.direction == "minimize" and cost < best_cost):
                    best_cost = cost
                    best_config = config

                costs.append(-cost if self.direction == "maximize" else cost)

            es.tell(solutions, costs)
            es.disp()

        return best_config