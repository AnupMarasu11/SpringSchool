from main import run

from src.costs import B1HomogeneityCost
from src.costs.b1_HomogeneitySARCost import B1HomogeneitySARCost_CUSTOM
from src.costs import B1HomogeneitySARCost
from src.data import Simulation
from src.utils import evaluate_coil_config

import numpy as np
import json

if __name__ == "__main__":
    # Load simulation data
    simulation = Simulation("src/data/simulations/children_1_tubes_6_id_23713.h5")
    
    # Define cost function
    #cost_function = B1HomogeneityCost()
    cost_function = B1HomogeneitySARCost()
    
    # Run optimization
    best_coil_config = run(simulation=simulation, cost_function=cost_function)
    
    # Evaluate best coil configuration
    result = evaluate_coil_config(best_coil_config, simulation, cost_function)

    # Save results to JSON file
    with open("results_SAR_400_CROP_Timeout.json", "w") as f:
        json.dump(result, f, indent=4)