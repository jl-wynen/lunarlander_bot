# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal, Union

import numpy as np

from lunarlander import Instructions, config
from lunarlander.tools import PlayerInfo, AsteroidInfo

g = config.gravity[1]


def log(*msg) -> None:
    print(f"Jankas: {' '.join(map(str, msg))}")


def rotate(
    current: float, target: float, tolerance: float = 0.5
) -> Union[Literal["left", "right"], None]:
    if abs(current - target) < tolerance:
        return
    return "left" if current < target else "right"


def wrapping_diff(target: float, current: float) -> float:
    diff1 = target - current
    diff2 = target + config.nx - current
    diff3 = target - config.nx - current
    diffs = [diff1, diff2, diff3]
    i = np.argmin(np.abs(diffs))
    return diffs[i]


def find_landing_site(terrain: np.ndarray) -> Union[int, None]:
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

    # Return location if large enough
    if (end - start) > 40:
        loc = int(start + (end - start) * 0.5)
        log("Found landing site at", loc)
        return loc


def sink_to_height(h0: float, v0: float, heading: float, ht: float) -> bool:
    a = np.cos(heading * np.pi / 180) * config.thrust
    g = abs(config.gravity[1])
    h1 = ((a - g) * ht + g * h0 + v0**2) / (a - 2 * g)
    log(h0, h1, ht)
    return h1 > h0


def stop_at_if_slow_down(h0: float, v0: float, heading: float) -> float:
    a = np.cos(heading * np.pi / 180) * config.thrust
    g = abs(config.gravity[1])
    return h0 - v0**2 / (a - g)


class Bot:
    """
    This is the lander-controlling bot that will be instantiated for the competition.
    """

    def __init__(self):
        self.team = "Jankas"  # This is your team name
        self.avatar = 15  # Optional attribute
        self.flag = "de"  # Optional attribute
        self.target_site = None
        self.state = self.initial_manoeuvre

    def run(
        self,
        t: float,
        dt: float,
        terrain: np.ndarray,
        players: dict[str, PlayerInfo],
        asteroids: list[AsteroidInfo],
    ):
        """
        This is the method that will be called at every time step to get the
        instructions for the ship.

        Parameters
        ----------
        t:
            The current time in seconds.
        dt:
            The time step in seconds.
        terrain:
            The (1d) array representing the lunar surface altitude.
        players:
            A dictionary of the players in the game. The keys are the team names and
            the values are the information about the players.
        asteroids:
            A list of the asteroids currently flying.
        """
        instructions, next_state = self.state(
            t=t, dt=dt, terrain=terrain, players=players, asteroids=asteroids
        )
        if next_state != self.state:
            log("Switching to", next_state.__name__)
        self.state = next_state
        return instructions

    def initial_manoeuvre(
        self, players: dict[str, PlayerInfo], **kwargs
    ) -> tuple[Instructions, callable]:
        me = players[self.team]
        vx, vy = me.velocity
        heading = me.heading
        instructions = Instructions()

        if vx > 10:
            instructions.main = True
        else:
            command = rotate(current=heading, target=0)
            if command == "left":
                instructions.left = True
            elif command == "right":
                instructions.right = True
            else:
                return instructions, self.wait_for_landing_site
        return instructions, self.initial_manoeuvre

    def wait_for_landing_site(
        self, players: dict[str, PlayerInfo], terrain: np.ndarray, **kwargs
    ) -> tuple[Instructions, callable]:
        instructions = Instructions()
        me = players[self.team]

        # Search for a suitable landing site
        if self.target_site is None:
            self.target_site = find_landing_site(terrain)
        if self.target_site is not None:
            return instructions, self.align_with_landing_site

        hover_height = np.max(terrain) + 50
        would_stop_at = stop_at_if_slow_down(
            h0=me.position[1], v0=me.velocity[1], heading=me.heading
        )
        if would_stop_at < hover_height and me.velocity[1] < 0:
            instructions.main = True
        return instructions, self.wait_for_landing_site

    def align_with_landing_site(
        self,
        players: dict[str, PlayerInfo],
        terrain: np.ndarray,
        **kwargs,
    ) -> tuple[Instructions, callable]:
        instructions = Instructions()
        me = players[self.team]
        x, y = me.position
        vx, vy = me.velocity
        heading = me.heading

        hover_height = np.max(terrain) + 50
        would_stop_at = stop_at_if_slow_down(
            h0=me.position[1], v0=me.velocity[1], heading=me.heading
        )
        if would_stop_at < hover_height and me.velocity[1] < 0:
            instructions.main = True

        diff = self.target_site - x
        if np.abs(diff) < 50:
            # Reduce horizontal speed
            if abs(vx) <= 0.1:
                command = rotate(current=heading, target=0)
            elif vx > 0.1:
                command = rotate(current=heading, target=90)
                instructions.main = True
            else:
                command = rotate(current=heading, target=-90)
                instructions.main = False

            if command == "left":
                instructions.left = True
            elif command == "right":
                instructions.right = True

            if (abs(vx) < 0.5) and (vy < -3):
                instructions.main = True
        if np.abs(diff) < 30:
            return instructions, self.land
        return instructions, self.align_with_landing_site

    def land(
        self, players: dict[str, PlayerInfo], terrain: np.ndarray, **kwargs
    ) -> tuple[Instructions, callable]:
        instructions = Instructions()
        me = players[self.team]
        x, y = me.position
        vx, vy = me.velocity
        heading = me.heading

        diff = self.target_site - x
        if np.abs(diff) < 50:
            # Reduce horizontal speed
            if abs(vx) <= 0.1:
                command = rotate(current=heading, target=0)
            elif vx > 0.1:
                command = rotate(current=heading, target=45)
                instructions.main = True
            else:
                command = rotate(current=heading, target=-45)
                instructions.main = False

            if command == "left":
                instructions.left = True
            elif command == "right":
                instructions.right = True

            if (abs(vx) > 2) and (vy < -3):
                instructions.main = True

        target_height = terrain[self.target_site]
        would_stop_at = stop_at_if_slow_down(h0=y, v0=vy, heading=heading)
        if would_stop_at < target_height - 60:
            instructions.main = True
        if abs(y - terrain[self.target_site]) < 15:
            instructions.main = True
        if vy < -5:
            instructions.main = True

        return instructions, self.land
