from .base import BaseCost
from ..data.simulation import SimulationData
from ..data.utils import B1Calculator

import numpy as np


class B1HomogeneitySARCost_CUSTOM(BaseCost):
    def __init__(self, lambda_weight: float = 0.01) -> None:
        super().__init__()
        self.direction = "maximize"
        self.lambda_weight = lambda_weight
        self.b1_calculator = B1Calculator()

    def calculate_cost(self, simulation_data: SimulationData) -> float:
        # B1+ Homogeneity component
        b1_field = self.b1_calculator(simulation_data)
        subject = simulation_data.subject

        b1_field_abs = np.abs(b1_field)
        b1_subject_voxels = b1_field_abs[subject]
        b1_score = np.mean(b1_subject_voxels) / (np.std(b1_subject_voxels) + 1e-8)

        # SAR component
        # Extract E-field: complex form (shape: 3D vector field)
        E_real = simulation_data.field[0, 0]  # Re(E)
        E_imag = simulation_data.field[0, 1]  # Im(E)
        E_complex = E_real + 1j * E_imag      # Shape: (3, x, y, z)

        E_magnitude_squared = np.sum(np.abs(E_complex) ** 2, axis=0)  # ||E||^2, shape: (x, y, z)

        sigma = simulation_data.properties[0]
        rho = simulation_data.properties[1]

        sar = E_magnitude_squared * sigma / (rho + 1e-8)
        max_sar = np.max(sar[subject])

        # Total cost
        return b1_score - self.lambda_weight * max_sar
