from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO

from src.envs.grid_world import SimpleGridWorldEnv


DEFAULT_MODEL = "dqn"
MODEL_PATH = Path("models") / "dqn_gridworld.zip"
PATH_PLOT_PATH = Path("plots") / "gridworld_path.png"
PATH_RESULTS_PATH = Path("results") / "gridworld_agent_path.csv"
MAX_PATH_STEPS = 50
MODEL_CLASSES = {
    "dqn": DQN,
    "ppo": PPO,
}


def create_env() -> SimpleGridWorldEnv:
    return SimpleGridWorldEnv()


def infer_model_key(model_path: Path) -> str:
    name = model_path.name.lower()
    if name.startswith("ppo_"):
        return "ppo"
    return "dqn"


def collect_agent_path(model, max_steps: int = MAX_PATH_STEPS) -> tuple[np.ndarray, float]:
    env = create_env()
    obs, info = env.reset()
    path = [info["position"]]
    total_reward = 0.0
    terminated = False
    truncated = False

    while not terminated and not truncated and len(path) <= max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        path.append(info["position"])

    env.close()
    return np.array(path), total_reward


def save_results(path: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"step": step, "row": int(position[0]), "col": int(position[1])}
        for step, position in enumerate(path)
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def plot_agent_path(path: np.ndarray, output_path: Path) -> Path:
    env = create_env()
    traps = np.array([tuple(trap) for trap in env.traps])

    plt.figure(figsize=(6, 6))
    plt.xlim(-0.5, env.size - 0.5)
    plt.ylim(env.size - 0.5, -0.5)
    plt.xticks(range(env.size))
    plt.yticks(range(env.size))
    plt.grid(True)

    plt.scatter(path[:, 1], path[:, 0], marker="o", label="Agent path")
    plt.plot(path[:, 1], path[:, 0], linewidth=2)
    plt.scatter([0], [0], marker="s", s=120, label="Start")
    plt.scatter([env.goal_position[1]], [env.goal_position[0]], marker="*", s=180, label="Goal")
    plt.scatter(traps[:, 1], traps[:, 0], marker="x", s=150, label="Traps")

    plt.title("GridWorld agent path")
    plt.legend(loc="upper right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    env.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a trained GridWorld agent.")
    parser.add_argument("--model", choices=MODEL_CLASSES.keys(), default=None)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=PATH_PLOT_PATH)
    parser.add_argument("--path-output", type=Path, default=PATH_RESULTS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_key = args.model or infer_model_key(args.model_path)
    model = MODEL_CLASSES[model_key].load(args.model_path)
    path, total_reward = collect_agent_path(model)
    path_results = save_results(path, args.path_output)
    path_plot = plot_agent_path(path, args.output)

    print(f"Saved path CSV to: {path_results}")
    print(f"Saved path plot to: {path_plot}")
    print(f"Path reward: {round(total_reward, 4)}")


if __name__ == "__main__":
    main()
