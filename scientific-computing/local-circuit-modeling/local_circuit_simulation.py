"""
Portfolio-safe local circuit simulation utilities.

This module keeps the reusable scientific-computing logic from a private
notebook: building connectivity matrices, simulating firing-rate dynamics, and
comparing observed network structure against shuffled controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    inhibitory_ratio: float = 0.15
    connectivity_gain: float = 0.002
    connectivity_probability: float = 0.25
    inhibitory_excitatory_balance: float = 0.15
    dt: float = 0.01
    tau: float = 0.25
    t_max: float = 50.0
    stimulus_start: float = 1.0
    stimulus_end: float = 15.0
    stimulus_strength: float = 10.0


@dataclass(frozen=True)
class RecoverySummary:
    label: str
    mean_seconds: float
    std_seconds: float
    samples: tuple[float, ...]


def filter_synapses(
    synapses: pd.DataFrame,
    pre_root_ids: Iterable[int] | None = None,
    post_root_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    pre_mask = np.ones(len(synapses), dtype=bool)
    post_mask = np.ones(len(synapses), dtype=bool)

    if pre_root_ids is not None:
        pre_mask = np.isin(synapses["pre_pt_root_id"], list(pre_root_ids))
    if post_root_ids is not None:
        post_mask = np.isin(synapses["post_pt_root_id"], list(post_root_ids))

    return synapses[pre_mask & post_mask]


def build_connectivity_matrix(
    synapses: pd.DataFrame,
    scale: float = 300.0,
) -> tuple[np.ndarray, np.ndarray]:
    matrix_frame = synapses.pivot_table(
        index="pre_pt_root_id",
        columns="post_pt_root_id",
        values="size",
        aggfunc="sum",
    ).fillna(0)
    shared_ids = matrix_frame.index[np.isin(matrix_frame.index, matrix_frame.columns)]
    square = matrix_frame.reindex(shared_ids).reindex(columns=shared_ids)
    return np.asarray(square, dtype=float) / scale, np.asarray(shared_ids)


def balance_inhibitory_input(
    weights: np.ndarray,
    n_excitatory: int,
    inhibitory_excitatory_balance: float,
) -> np.ndarray:
    balanced = weights.copy()
    excitatory_input = balanced[:n_excitatory].sum(axis=0)
    inhibitory_input = balanced[n_excitatory:].sum(axis=0)
    safe_inhibitory = np.where(inhibitory_input == 0, 1.0, inhibitory_input)
    scaling = inhibitory_excitatory_balance * excitatory_input / safe_inhibitory
    balanced[n_excitatory:] *= scaling
    balanced[n_excitatory:] *= -1
    return balanced


def shuffle_connections(weights: np.ndarray, random_seed: int) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    shuffled = weights.copy().reshape(-1)
    rng.shuffle(shuffled)
    return shuffled.reshape(weights.shape)


def add_random_inhibitory_population(
    excitatory_weights: np.ndarray,
    config: SimulationConfig,
    random_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    n_excitatory = excitatory_weights.shape[0]
    n_total = int(n_excitatory / (1 - config.inhibitory_ratio))
    weights = np.zeros((n_total, n_total), dtype=float)
    weights[:n_excitatory, :n_excitatory] = excitatory_weights

    random_mask = rng.random((n_total - n_excitatory, n_total)) < config.connectivity_probability
    random_weights = rng.exponential(scale=1.0, size=random_mask.shape) * random_mask
    weights[n_excitatory:, :] = random_weights

    weights *= config.connectivity_gain
    return balance_inhibitory_input(
        weights,
        n_excitatory=n_excitatory,
        inhibitory_excitatory_balance=config.inhibitory_excitatory_balance,
    )


def generate_stimulus(n_neurons: int, config: SimulationConfig) -> np.ndarray:
    n_steps = int(config.t_max / config.dt)
    n_excitatory = int(n_neurons * (1 - config.inhibitory_ratio))
    stimulus = np.zeros((n_neurons, n_steps), dtype=float)
    t = np.arange(n_steps) * config.dt
    mask = (t >= config.stimulus_start) & (t <= config.stimulus_end)
    stimulus[:n_excitatory, mask] = config.stimulus_strength
    return stimulus


def transfer(values: np.ndarray, activation: str = "relu") -> np.ndarray:
    if activation == "linear":
        return values
    if activation == "relu":
        return np.maximum(0, values)
    raise ValueError(f"Unsupported activation: {activation}")


def run_simulation(
    weights: np.ndarray,
    stimulus: np.ndarray,
    config: SimulationConfig,
    activation: str = "relu",
) -> np.ndarray:
    n_neurons, n_steps = stimulus.shape
    rates = np.zeros((n_neurons, n_steps), dtype=float)

    for step in range(1, n_steps):
        recurrent_input = weights.T @ rates[:, step - 1]
        drive = transfer(recurrent_input + stimulus[:, step], activation)
        rates[:, step] = rates[:, step - 1] + (config.dt / config.tau) * (
            -rates[:, step - 1] + drive
        )

    return rates


def recovery_time(
    rates: np.ndarray,
    stimulus: np.ndarray,
    config: SimulationConfig,
    threshold_ratio: float = 0.01,
) -> float:
    stimulus_mask = stimulus[0] > 0
    stimulus_start = np.where(stimulus_mask)[0][0]
    stimulus_end = np.where(stimulus_mask)[0][-1]
    baseline = np.mean(rates[:, stimulus_start - 1])
    threshold = np.mean(stimulus[0][stimulus_mask]) * threshold_ratio + baseline
    post_stimulus = np.mean(rates[:, stimulus_end:], axis=0)
    recovered = np.where(post_stimulus < threshold)[0]
    if len(recovered) == 0:
        raise ValueError("Activity did not recover within the simulation window.")
    return recovered[0] * config.dt


def count_reciprocal_connections(weights: np.ndarray) -> int:
    reciprocal_mask = (weights > 0) & (weights.T > 0)
    return int(np.sum(reciprocal_mask) // 2)


def remove_reciprocal_connections(weights: np.ndarray) -> np.ndarray:
    result = weights.copy()
    rows, cols = np.where((result > 0) & (result.T > 0))
    for row, col in zip(rows, cols):
        if row < col:
            result[col, row] = 0
    return result


def compare_observed_to_shuffled(
    observed_excitatory_weights: np.ndarray,
    seeds: Iterable[int],
    config: SimulationConfig,
) -> list[RecoverySummary]:
    n_excitatory = observed_excitatory_weights.shape[0]
    n_total = int(n_excitatory / (1 - config.inhibitory_ratio))
    stimulus = generate_stimulus(n_total, config)

    samples: dict[str, list[float]] = {"observed": [], "shuffled": []}
    for seed in seeds:
        for label, excitatory_weights in (
            ("observed", observed_excitatory_weights),
            ("shuffled", shuffle_connections(observed_excitatory_weights, seed)),
        ):
            weights = add_random_inhibitory_population(excitatory_weights, config, seed)
            rates = run_simulation(weights, stimulus, config)
            samples[label].append(recovery_time(rates[:n_excitatory], stimulus, config))

    return [
        RecoverySummary(
            label=label,
            mean_seconds=float(np.mean(values)),
            std_seconds=float(np.std(values)),
            samples=tuple(float(value) for value in values),
        )
        for label, values in samples.items()
    ]
