from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("results") / "gridworld_dqn_rewards.csv"
REWARD_PLOT_PATH = Path("plots") / "gridworld_rewards.png"


def plot_rewards(
    input_path: Path = RESULTS_PATH,
    output_path: Path = REWARD_PLOT_PATH,
    title: str | None = None,
    rolling_window: int = 1,
) -> Path:
    results = pd.read_csv(input_path)
    x_column = "timestep" if "timestep" in results.columns else "episode"

    plt.figure(figsize=(7, 4))
    plt.plot(results[x_column], results["reward"], linewidth=1.5, alpha=0.45, label="Episode reward")
    if rolling_window > 1:
        rolling_rewards = results["reward"].rolling(rolling_window).mean()
        plt.plot(
            results[x_column],
            rolling_rewards,
            linewidth=2.5,
            label=f"Rolling mean ({rolling_window})",
        )
        plt.legend()
    plt.xlabel("Timestep" if x_column == "timestep" else "Episode")
    plt.ylabel("Reward")
    plt.title(title or "GridWorld rewards")
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GridWorld evaluation rewards from CSV.")
    parser.add_argument("--input", type=Path, default=RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=REWARD_PLOT_PATH)
    parser.add_argument("--title", default=None)
    parser.add_argument("--rolling-window", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = plot_rewards(
        args.input,
        args.output,
        args.title,
        args.rolling_window,
    )
    print(f"Saved reward plot to: {output_path}")


if __name__ == "__main__":
    main()
