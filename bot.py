# SPDX-License-Identifier: BSD-3-Clause

# flake8: noqa F401

import numpy as np

from lunarlander import Instructions

#     Checkpoint,
#     Heading,
# ,
#     Location,
#     MapProxy,
#     Vector,
#     WeatherForecast,
#     config,
# )
# from vendeeglobe.utils import distance_on_surface

CREATOR = "TeamName"  # This is your team name


class Bot:
    """
    This is the ship-controlling bot that will be instantiated for the competition.
    """

    def __init__(self):
        self.team = CREATOR  # Mandatory attribute
        self.initial_rotation_complete = False

    def run(
        self,
        t: float,
        dt: float,
        x: float,
        y: float,
        heading: float,
        vx: float,
        vy: float,
        fuel: float,
        terrain: np.ndarray,
    ):
        """
        This is the method that will be called at every time step to get the
        instructions for the ship.

        Parameters
        ----------
        t:
            The current time in hours.
        dt:
            The time step in hours.
        """
        instructions = Instructions()
        if (not self.initial_rotation_complete) and (heading > 0):
            instructions.right = True
        if (not self.initial_rotation_complete) and (heading <= 2):
            instructions.right = False
            self.initial_rotation_complete = True

        if not self.initial_rotation_complete:
            return instructions

        if y < 900:
            instructions.main = True

        return instructions

    def find_landing_site(self, terrain):
        gradient = np.gradient(terrain)
