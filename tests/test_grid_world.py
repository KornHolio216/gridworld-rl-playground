import numpy as np
from stable_baselines3.common.env_checker import check_env

from src.envs.grid_world import SimpleGridWorldEnv


def test_reset_returns_expected_observation_format():
    env = SimpleGridWorldEnv()
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == (2,)
    assert obs.tolist() == [0.0, 0.0]
    assert info["position"] == (0, 0)


def test_agent_moves_inside_grid():
    env = SimpleGridWorldEnv()
    env.reset()

    obs, reward, terminated, truncated, info = env.step(3)

    assert obs.tolist() == [0.0, 0.25]
    assert reward == -0.01
    assert not terminated
    assert not truncated
    assert info["position"] == (0, 1)


def test_goal_gives_positive_reward():
    env = SimpleGridWorldEnv()
    env.reset()
    env.agent_position = env.goal_position.copy()
    env.agent_position[1] -= 1

    _, reward, terminated, truncated, info = env.step(3)

    assert reward == 1.0
    assert terminated
    assert not truncated
    assert info["is_goal"]


def test_trap_gives_negative_reward():
    env = SimpleGridWorldEnv()
    env.reset()
    env.agent_position = env.traps[0].copy()
    env.agent_position[1] -= 1

    _, reward, terminated, truncated, info = env.step(3)

    assert reward == -1.0
    assert terminated
    assert not truncated
    assert info["is_trap"]


def test_step_limit_sets_truncated():
    env = SimpleGridWorldEnv(size=5, traps=[], max_steps=1)
    env.reset()

    _, reward, terminated, truncated, info = env.step(0)

    assert reward == -0.01
    assert not terminated
    assert truncated
    assert info["steps"] == 1


def test_random_traps_are_seeded_and_skip_start_and_goal():
    first_env = SimpleGridWorldEnv(random_traps=True, trap_count=4, trap_seed=123)
    second_env = SimpleGridWorldEnv(random_traps=True, trap_count=4, trap_seed=123)

    first_traps = [tuple(int(value) for value in trap) for trap in first_env.traps]
    second_traps = [tuple(int(value) for value in trap) for trap in second_env.traps]

    assert first_traps == second_traps
    assert len(first_traps) == 4
    assert (0, 0) not in first_traps
    assert (4, 4) not in first_traps


def test_invalid_grid_size_raises_error():
    try:
        SimpleGridWorldEnv(size=1)
    except ValueError as error:
        assert "Grid size" in str(error)
    else:
        raise AssertionError("Expected ValueError for grid size below 2.")


def test_invalid_trap_position_raises_error():
    try:
        SimpleGridWorldEnv(traps=[(5, 0)])
    except ValueError as error:
        assert "outside the grid" in str(error)
    else:
        raise AssertionError("Expected ValueError for trap outside the grid.")


def test_trap_cannot_overlap_start_or_goal():
    for trap in [(0, 0), (4, 4)]:
        try:
            SimpleGridWorldEnv(traps=[trap])
        except ValueError as error:
            assert "cannot be a trap" in str(error)
        else:
            raise AssertionError("Expected ValueError for trap on start or goal.")


def test_duplicate_traps_raise_error():
    try:
        SimpleGridWorldEnv(traps=[(1, 2), (1, 2)])
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("Expected ValueError for duplicate traps.")


def test_max_steps_must_be_positive():
    for max_steps in [0, -1]:
        try:
            SimpleGridWorldEnv(max_steps=max_steps)
        except ValueError as error:
            assert "Max steps" in str(error)
        else:
            raise AssertionError("Expected ValueError for non-positive max_steps.")


def test_environment_passes_stable_baselines_checker():
    env = SimpleGridWorldEnv()

    check_env(env, warn=True)
