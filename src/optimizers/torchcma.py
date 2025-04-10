import torch
import numpy as np
import cma
from ..data.simulation import Simulation, SimulationData, CoilConfig
import time

class TorchCMAOptimizer:
    def __init__(self, cost_function, direction="maximize", sigma=0.3, max_iter=100):
        self.cost_function = cost_function
        self.direction = direction
        self.sigma = sigma
        self.max_iter = max_iter
    """
    def optimize(self, simulation):
        # Initial vector: 8 amplitudes [0,1], 8 phases [0, 2π]
        x0 = np.concatenate([
            np.random.uniform(low=0.0, high=1.0, size=8),
            np.random.uniform(low=0.0, high=2 * np.pi, size=8)
        ])

        # Define CMA optimizer with bounds
        es = cma.CMAEvolutionStrategy(x0, self.sigma, {
            'maxiter': self.max_iter,
            'bounds': [np.concatenate([np.zeros(8), np.zeros(8)]),
                       np.concatenate([np.ones(8), 2*np.pi*np.ones(8)])]
        })

        best_cost = -float('inf') if self.direction == "maximize" else float('inf')
        best_config = None

        while not es.stop():
            solutions = es.ask()
            costs = []

            for x in solutions:
                x_tensor = torch.tensor(x, dtype=torch.float32)
                config = self._apply_parameters(x_tensor)

                # Assume simulation and cost_function return a scalar (or tensor with .item())
                sim_data = simulation(config)
                cost = self.cost_function(sim_data)

                cost_val = cost.item() if isinstance(cost, torch.Tensor) else cost

                if (self.direction == "maximize" and cost_val > best_cost) or \
                   (self.direction == "minimize" and cost_val < best_cost):
                    best_cost = cost_val
                    best_config = config

                costs.append(-cost_val if self.direction == "maximize" else cost_val)

            es.tell(solutions, costs)
            es.disp()

        return best_config
    """
    def _apply_parameters(self, x: np.ndarray) -> CoilConfig:
        amplitude = x[:8]
        phase = x[8:]
        return CoilConfig(phase=phase, amplitude=amplitude)
    
    def optimize(self, simulation):
        # Initial vector: 8 amplitudes [0,1], 8 phases [0, 2π]
        x0 = np.concatenate([
            np.random.uniform(low=0.0, high=1.0, size=8),
            np.random.uniform(low=0.0, high=2 * np.pi, size=8)
        ])

        # CMA optimizer with bounds
        es = cma.CMAEvolutionStrategy(x0, self.sigma, {
            'maxiter': self.max_iter,
            'bounds': [
                np.concatenate([np.zeros(8), np.zeros(8)]),
                np.concatenate([np.ones(8), 2 * np.pi * np.ones(8)])
            ]
        })

        best_cost = -float('inf') if self.direction == "maximize" else float('inf')
        best_config = None

        timeout = 270  # 4 minutes 30 seconds
        start_time = time.time()

        while not es.stop():
            # Check timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                print("Optimization timed out. Returning best result so far.")
                break

            solutions = es.ask()
            costs = []

            for x in solutions:
                x_tensor = torch.tensor(x, dtype=torch.float32)
                config = self._apply_parameters(x_tensor)

                sim_data = simulation(config)
                cost = self.cost_function(sim_data)

                cost_val = cost.item() if isinstance(cost, torch.Tensor) else cost

                if (self.direction == "maximize" and cost_val > best_cost) or \
                (self.direction == "minimize" and cost_val < best_cost):
                    best_cost = cost_val
                    best_config = config

                costs.append(-cost_val if self.direction == "maximize" else cost_val)

            es.tell(solutions, costs)
            es.disp()

        return best_config