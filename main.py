from src.costs.base import BaseCost
from src.optimizers import DummyOptimizer
from src.optimizers.cma import CMAESOptimizer
from src.optimizers.torchcma import TorchCMAOptimizer
from src.data import Simulation, CoilConfig
from src.costs import B1HomogeneityCost

import numpy as np

def run(simulation: Simulation, 
        cost_function: BaseCost,
        timeout: int = 100) -> CoilConfig:
    """
        Main function to run the optimization, returns the best coil configuration

        Args:
            simulation: Simulation object
            cost_function: Cost function object
            timeout: Time (in seconds) after which the evaluation script will be terminated
    """
    #optimizer = DummyOptimizer(cost_function=cost_function)
    #best_coil_config = optimizer.optimize(simulation)

    optimizer = TorchCMAOptimizer(cost_function=cost_function, sigma=0.3, max_iter=500)
    best_config = optimizer.optimize(simulation)

    return best_config