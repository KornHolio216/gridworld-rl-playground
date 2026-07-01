from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import pandas as pd
from gymnasium.error import DependencyNotInstalled
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.envs.grid_world import SimpleGridWorldEnv


RANDOM_SEED = 42
DEFAULT_ENV = "gridworld"
DEFAULT_MODEL = "dqn"
DEFAULT_TIMESTEPS = 30_000
MODEL_PATH = Path("models") / "dqn_gridworld.zip"

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


class TrainingRewardCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, int | float]] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is None:
                continue

            self.rows.append(
                {
                    "episode": len(self.rows) + 1,
                    "timestep": self.num_timesteps,
                    "reward": round(float(episode["r"]), 4),
                    "steps": int(episode["l"]),
                }
            )
        return True


def create_env(env_key: str = DEFAULT_ENV) -> Monitor:
    if env_key == "gridworld":
        return Monitor(SimpleGridWorldEnv())
    if env_key == "frozenlake":
        return Monitor(gym.make("FrozenLake-v1", is_slippery=False))
    try:
        return Monitor(gym.make(ENV_IDS[env_key]))
    except DependencyNotInstalled as error:
        message = (
            f"{ENV_IDS[env_key]} requires an optional Gymnasium dependency. "
            'For LunarLander, install it with: pip install swig "gymnasium[box2d]"'
        )
        raise RuntimeError(message) from error


def default_model_path(env_key: str, model_key: str) -> Path:
    return Path("models") / f"{model_key}_{env_key}.zip"


def default_training_results_path(env_key: str, model_key: str) -> Path:
    return Path("results") / f"{env_key}_{model_key}_training_rewards.csv"


def create_model(model_key: str, env_key: str, env: Monitor):
    model_class = MODEL_CLASSES[model_key]

    if env_key == "gridworld" and model_key == "dqn":
        return model_class(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            buffer_size=10_000,
            learning_starts=100,
            batch_size=32,
            gamma=0.95,
            train_freq=4,
            target_update_interval=250,
            exploration_fraction=0.4,
            exploration_final_eps=0.05,
            seed=RANDOM_SEED,
            verbose=0,
        )

    if env_key == "gridworld" and model_key == "ppo":
        return model_class(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=64,
            gamma=0.95,
            seed=RANDOM_SEED,
            verbose=0,
        )

    return model_class("MlpPolicy", env, seed=RANDOM_SEED, verbose=0)


def train_agent(
    env_key: str,
    model_key: str,
    timesteps: int,
    model_path: Path | None = None,
    training_output: Path | None = None,
) -> tuple[Path, Path]:
    env = create_env(env_key)
    model = create_model(model_key, env_key, env)
    callback = TrainingRewardCallback()

    model.learn(total_timesteps=timesteps, callback=callback)

    output_path = model_path or default_model_path(env_key, model_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    training_output_path = training_output or default_training_results_path(env_key, model_key)
    training_output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(callback.rows).to_csv(training_output_path, index=False)
    env.close()

    return output_path, training_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RL agent.")
    parser.add_argument("--env", choices=ENV_IDS.keys(), default=DEFAULT_ENV)
    parser.add_argument("--model", choices=MODEL_CLASSES.keys(), default=DEFAULT_MODEL)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--training-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        model_path, training_output = train_agent(
            args.env,
            args.model,
            args.timesteps,
            args.model_path,
            args.training_output,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    print(f"Saved model to: {model_path}")
    print(f"Saved training rewards to: {training_output}")


if __name__ == "__main__":
    main()
