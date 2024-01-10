# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from lunarlander import Instructions


CREATOR = "Apollo 11"  # This is your team name


class Bot:
    """
    This is the lander-controlling bot that will be instantiated for the competition.
    """

    def __init__(self):
        self.team = CREATOR  # Mandatory attribute
        self.initial_manoeuvre = True
        self.target_site = None
        self.target_heading = 0

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

        if self.initial_manoeuvre:
            if vx > 10:
                instructions.main = True
            elif self.target_heading is not None:
                command = self.rotate(heading)
                if command == "left":
                    instructions.left = True
                elif command == "right":
                    instructions.right = True
            if self.target_heading is None:
                self.initial_manoeuvre = False
            return instructions

        if self.target_site is None:
            self.target_site = self.find_landing_site(terrain)

        if (self.target_site is None) and (y < 900) and (vy < 0):
            instructions.main = True

        if self.target_site is not None:
            command = None
            diff = self.target_site - x
            if np.abs(diff) < 50:
                if vx > 0.1:
                    self.target_heading = 90
                    command = self.rotate(heading)
                    instructions.main = True
                else:
                    self.target_heading = 0
                    command = self.rotate(heading)
                    instructions.main = False

                if command == "left":
                    instructions.left = True
                elif command == "right":
                    instructions.right = True

                if (abs(vx) < 0.5) and (vy < -3):
                    instructions.main = True
            else:
                if vy < 0:
                    instructions.main = True

        if t < 5:
            instructions.main = True

        return instructions

    def rotate(self, heading):
        if np.abs(heading - self.target_heading) < 0.35:
            self.target_heading = None
            return
        if heading < self.target_heading:
            return "left"
        else:
            return "right"

    def find_landing_site(self, terrain):
        # Find largest landing site
        n = len(terrain)
        # Find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(terrain[:-1], terrain[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # Find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        # Find largest run
        imax = np.argmax(run_lengths)
        start = run_starts[imax]
        end = start + run_lengths[imax]

        if (end - start) > 40:
            loc = int(start + (end - start) * 0.5)
            print("Found landing site at", loc)
            return loc
