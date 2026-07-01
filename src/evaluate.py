from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.error import DependencyNotInstalled
from stable_baselines3 import DQN, PPO

from src.envs.grid_world import SimpleGridWorldEnv


DEFAULT_ENV = "gridworld"
DEFAULT_MODEL = "dqn"
MODEL_PATH = Path("models") / "dqn_gridworld.zip"
RESULTS_PATH = Path("results") / "gridworld_dqn_rewards.csv"
DEFAULT_EPISODES = 10

ENV_IDS = {
    "gridworld": "SimpleGridWorldEnv",
    "frozenlake": "FrozenLake-v1",
    "acrobot": "Acrobot-v1",
    "lunarlander": "LunarLander-v3",
}
MODEL_CLASSES = {
    "dqn": DQN,
    "ppo": PPO,
}


def create_env(env_key: str = DEFAULT_ENV):
    if env_key == "gridworld":
        return SimpleGridWorldEnv()
    if env_key == "frozenlake":
        return gym.make("FrozenLake-v1", is_slippery=False)
    try:
        return gym.make(ENV_IDS[env_key])
    except DependencyNotInstalled as error:
        message = (
            f"{ENV_IDS[env_key]} requires an optional Gymnasium dependency. "
            'For LunarLander, install it with: pip install swig "gymnasium[box2d]"'
        )
        raise RuntimeError(message) from error


def default_results_path(env_key: str, model_key: str) -> Path:
    return Path("results") / f"{env_key}_{model_key}_rewards.csv"


def infer_model_key(model_path: Path) -> str:
    name = model_path.name.lower()
    if name.startswith("ppo_"):
        return "ppo"
    return "dqn"


def shortest_path_length(env: SimpleGridWorldEnv) -> int | None:
    start = tuple(int(value) for value in env.start_position)
    goal = tuple(int(value) for value in env.goal_position)
    traps = {tuple(int(value) for value in trap) for trap in env.traps}
    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        (row, col), distance = queue.popleft()
        if (row, col) == goal:
            return distance

        for move in SimpleGridWorldEnv.ACTIONS.values():
            next_row = min(max(row + int(move[0]), 0), env.size - 1)
            next_col = min(max(col + int(move[1]), 0), env.size - 1)
            next_cell = (next_row, next_col)

            if next_cell in visited or next_cell in traps:
                continue

            visited.add(next_cell)
            queue.append((next_cell, distance + 1))

    return None


def evaluate_agent(
    model,
    env_key: str,
    episodes: int,
) -> tuple[list[dict[str, float | int | bool]], dict[str, float]]:
    rows: list[dict[str, float | int | bool]] = []

    for episode in range(1, episodes + 1):
        env = create_env(env_key)
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

        success = bool(info.get("is_goal", False)) if env_key == "gridworld" else bool(total_reward > 0)
        env.close()
        rows.append(
            {
                "episode": episode,
                "reward": round(total_reward, 4),
                "steps": steps,
                "success": success,
            }
        )

    rewards = [float(row["reward"]) for row in rows]
    step_counts = [int(row["steps"]) for row in rows]
    successful_steps = [int(row["steps"]) for row in rows if row["success"]]
    success_rate = 100.0 * len(successful_steps) / len(rows)
    optimal_path_length = (
        shortest_path_length(SimpleGridWorldEnv())
        if env_key == "gridworld"
        else None
    )
    summary = {
        "mean_reward": round(float(np.mean(rewards)), 4),
        "min_reward": round(float(np.min(rewards)), 4),
        "max_reward": round(float(np.max(rewards)), 4),
        "mean_steps": round(float(np.mean(step_counts)), 2),
        "success_rate": round(success_rate, 2),
        "learned_path_length": round(float(np.mean(successful_steps)), 2)
        if successful_steps
        else float("nan"),
    }
    if optimal_path_length is not None:
        summary["optimal_path_length"] = float(optimal_path_length)
    return rows, summary


def save_results(rows: list[dict[str, float | int | bool]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent.")
    parser.add_argument("--env", choices=ENV_IDS.keys(), default=DEFAULT_ENV)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--model", choices=MODEL_CLASSES.keys(), default=None)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_key = args.model or infer_model_key(args.model_path)
    model = MODEL_CLASSES[model_key].load(args.model_path)
    try:
        rows, summary = evaluate_agent(model, args.env, args.episodes)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    output_path = save_results(
        rows,
        args.output or default_results_path(args.env, model_key),
    )

    print(f"Saved episode results to: {output_path}")
    print(f"Mean reward: {summary['mean_reward']}")
    print(f"Mean steps: {summary['mean_steps']}")
    print(f"Success rate: {summary['success_rate']}%")
    if "optimal_path_length" in summary:
        print(f"Optimal path length: {int(summary['optimal_path_length'])}")
        print(f"Learned path length: {summary['learned_path_length']}")
    print(f"Min reward: {summary['min_reward']}")
    print(f"Max reward: {summary['max_reward']}")


if __name__ == "__main__":
    main()
