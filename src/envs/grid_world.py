from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


DEFAULT_GRID_SIZE = 5
DEFAULT_TRAPS = [(1, 2), (2, 2), (3, 1)]
GOAL_REWARD = 1.0
TRAP_REWARD = -1.0
STEP_PENALTY = -0.01


class SimpleGridWorldEnv(gym.Env):
    """Small custom GridWorld environment compatible with Gymnasium.

    Reward design:
    - +1.0 for reaching the goal,
    - -1.0 for entering a trap,
    - -0.01 for every regular move.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    ACTIONS = {
        0: np.array([-1, 0], dtype=np.int32),  # up
        1: np.array([1, 0], dtype=np.int32),   # down
        2: np.array([0, -1], dtype=np.int32),  # left
        3: np.array([0, 1], dtype=np.int32),   # right
    }

    def __init__(
        self,
        size: int = DEFAULT_GRID_SIZE,
        traps: list[tuple[int, int]] | None = None,
        random_traps: bool = False,
        trap_count: int = 3,
        trap_seed: int | None = None,
        max_steps: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.size = int(size)
        if self.size < 2:
            raise ValueError("Grid size must be at least 2.")

        self.start_position = np.array([0, 0], dtype=np.int32)
        self.goal_position = np.array([self.size - 1, self.size - 1], dtype=np.int32)
        trap_positions = self._create_traps(traps, random_traps, trap_count, trap_seed)
        self._validate_traps(trap_positions)
        self.traps = [np.array(trap, dtype=np.int32) for trap in trap_positions]
        self.max_steps = int(max_steps) if max_steps is not None else self.size * self.size * 2
        if self.max_steps <= 0:
            raise ValueError("Max steps must be positive.")
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.agent_position = self.start_position.copy()
        self.steps = 0

    def reset(self, seed=None, options=None):
        """Reset the agent to the start and return ``(observation, info)``."""
        super().reset(seed=seed)
        self.agent_position = self.start_position.copy()
        self.steps = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        move = self.ACTIONS[action]
        self.agent_position = np.clip(
            self.agent_position + move,
            0,
            self.size - 1,
        )
        self.steps += 1

        reward = STEP_PENALTY
        terminated = False
        truncated = self.steps >= self.max_steps

        if self._is_goal(self.agent_position):
            reward = GOAL_REWARD
            terminated = True
        elif self._is_trap(self.agent_position):
            reward = TRAP_REWARD
            terminated = True

        truncated = bool(truncated and not terminated)
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]

        for trap in self.traps:
            row, col = trap
            grid[row][col] = "X"

        goal_row, goal_col = self.goal_position
        agent_row, agent_col = self.agent_position
        grid[goal_row][goal_col] = "G"
        grid[agent_row][agent_col] = "A"

        output = "\n".join(" ".join(row) for row in grid)

        if self.render_mode == "human":
            print(output)

        return output

    def _get_obs(self):
        return (self.agent_position / (self.size - 1)).astype(np.float32)

    def _get_info(self):
        return {
            "position": tuple(int(value) for value in self.agent_position),
            "steps": self.steps,
            "is_goal": self._is_goal(self.agent_position),
            "is_trap": self._is_trap(self.agent_position),
        }

    def _is_goal(self, position):
        return bool(np.array_equal(position, self.goal_position))

    def _is_trap(self, position):
        return any(np.array_equal(position, trap) for trap in self.traps)

    def _create_traps(
        self,
        traps: list[tuple[int, int]] | None,
        random_traps: bool,
        trap_count: int,
        trap_seed: int | None,
    ) -> list[tuple[int, int]]:
        if traps is not None:
            return traps

        if not random_traps:
            return DEFAULT_TRAPS

        if trap_count < 0:
            raise ValueError("Trap count cannot be negative.")

        blocked = {
            tuple(int(value) for value in self.start_position),
            tuple(int(value) for value in self.goal_position),
        }
        cells = [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if (row, col) not in blocked
        ]

        trap_count = min(int(trap_count), len(cells))
        rng = np.random.default_rng(trap_seed)
        selected = rng.choice(len(cells), size=trap_count, replace=False)
        return [cells[int(index)] for index in selected]

    def _validate_traps(self, traps: list[tuple[int, int]]) -> None:
        start = tuple(int(value) for value in self.start_position)
        goal = tuple(int(value) for value in self.goal_position)
        if len(set(traps)) != len(traps):
            raise ValueError("Trap positions must be unique.")

        for trap in traps:
            row, col = trap
            if not (0 <= row < self.size and 0 <= col < self.size):
                raise ValueError(f"Trap {trap} is outside the grid.")
            if trap == start:
                raise ValueError("Start position cannot be a trap.")
            if trap == goal:
                raise ValueError("Goal position cannot be a trap.")
