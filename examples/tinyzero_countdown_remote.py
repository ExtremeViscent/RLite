"""Minimal TinyZero-style countdown loop for local Megatron + remote SGLang.

Drop this into a Megatron training script and fill in two hooks:

`get_source_snapshots()`
    Return the current local Megatron snapshots, one per training rank.

`actor_step(batch)`
    Run one local actor update. Each batch item already contains:
    `prompt`, `completion`, `old_token_logprobs`, `old_logprob`, `reward`, `advantage`.

The demo keeps the RL side intentionally small:
prompt -> remote SGLang rollout -> rule reward -> group-normalized advantage
-> local Megatron actor step -> tp4 -> tp2+dp2 weight sync.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import requests

from rlite.integrations import RemoteTopology, sync_megatron_to_remote_sglang
from rlite.weight_mapping import get_profile


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
ALLOWED_EQ_RE = re.compile(r"^[\d+\-*/().\s]+$")


@dataclass(frozen=True)
class CountdownSample:
    target: int
    nums: tuple[int, ...]


def make_countdown_prompt(sample: CountdownSample) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant. Think briefly, then return the final equation in "
        "<answer> </answer> tags.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Using the numbers {list(sample.nums)}, create an equation that equals {sample.target}. "
        "You can use +, -, *, / and each number can only be used once. "
        "Return only the final equation inside <answer> </answer> tags.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def make_countdown_batch(
    batch_size: int,
    *,
    rng: random.Random,
    num_operands: int = 4,
    min_number: int = 1,
    max_number: int = 13,
    max_target: int = 48,
) -> list[CountdownSample]:
    return [
        CountdownSample(
            target=rng.randint(1, max_target),
            nums=tuple(rng.randint(min_number, max_number) for _ in range(num_operands)),
        )
        for _ in range(batch_size)
    ]


def extract_equation(text: str) -> str | None:
    matches = list(ANSWER_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def validate_equation(equation: str, nums: tuple[int, ...]) -> bool:
    try:
        used = sorted(int(value) for value in re.findall(r"\d+", equation))
    except Exception:
        return False
    return used == sorted(nums)


def evaluate_equation(equation: str) -> float | None:
    if not ALLOWED_EQ_RE.match(equation):
        return None
    try:
        return float(eval(equation, {"__builtins__": None}, {}))
    except Exception:
        return None


def countdown_reward(text: str, sample: CountdownSample, *, format_score: float = 0.1) -> float:
    equation = extract_equation(text)
    if equation is None:
        return 0.0
    if not validate_equation(equation, sample.nums):
        return format_score
    value = evaluate_equation(equation)
    if value is None:
        return format_score
    return 1.0 if abs(value - sample.target) < 1e-5 else format_score


def rollout_prompt(
    remote_url: str,
    rollout_model: str,
    prompt: str,
    *,
    group_size: int,
    max_tokens: int = 96,
    temperature: float = 1.0,
) -> list[dict[str, object]]:
    response = requests.post(
        f"{remote_url.rstrip('/')}/v1/completions",
        json={
            "model": rollout_model,
            "prompt": prompt,
            "n": group_size,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "logprobs": 1,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    return [
        {
            "text": choice.get("text", ""),
            "token_logprobs": tuple(
                value for value in (choice.get("logprobs") or {}).get("token_logprobs", ()) if value is not None
            ),
        }
        for choice in payload["choices"]
    ]


def build_grpo_batch(
    samples: list[CountdownSample],
    prompts: list[str],
    grouped_rollouts: list[list[dict[str, object]]],
) -> list[dict[str, object]]:
    batch = []
    for sample, prompt, group in zip(samples, prompts, grouped_rollouts):
        rewards = [countdown_reward(item["text"], sample) for item in group]
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
        scale = math.sqrt(variance) + 1e-6
        for item, reward in zip(group, rewards):
            token_logprobs = tuple(float(value) for value in item["token_logprobs"])
            batch.append(
                {
                    "prompt": prompt,
                    "completion": item["text"],
                    "old_token_logprobs": token_logprobs,
                    "old_logprob": sum(token_logprobs),
                    "reward": reward,
                    "advantage": (reward - mean_reward) / scale,
                    "target": sample.target,
                    "nums": sample.nums,
                }
            )
    return batch


def default_rollout_topology() -> RemoteTopology:
    return RemoteTopology.from_grid(
        framework="sglang",
        tp_size=2,
        dp_size=2,
        rank_offset=4,
    )


def run_countdown_loop(
    *,
    get_source_snapshots: Callable[[], Iterable[object]],
    actor_step: Callable[[list[dict[str, object]]], Mapping[str, object] | None],
    remote_url: str,
    rollout_model: str,
    steps: int,
    batch_size: int = 8,
    group_size: int = 4,
    seed: int = 0,
    train_profile=None,
    rollout_profile=None,
    rollout_topology: RemoteTopology | None = None,
    sync_before_start: bool = True,
):
    train_profile = train_profile or get_profile(
        "qwen",
        "qwen2_5",
        overrides={"tensor_parallel_size": 4},
    )
    rollout_profile = rollout_profile or get_profile(
        "qwen",
        "qwen2_5",
        overrides={"tensor_parallel_size": 2},
    )
    rollout_topology = rollout_topology or default_rollout_topology()
    rng = random.Random(seed)

    def sync_weights():
        return sync_megatron_to_remote_sglang(
            tuple(get_source_snapshots()),
            train_profile=train_profile,
            rollout_profile=rollout_profile,
            topology=rollout_topology,
            remote_url=remote_url,
        )

    if sync_before_start:
        sync_weights()

    for step in range(steps):
        samples = make_countdown_batch(batch_size, rng=rng)
        prompts = [make_countdown_prompt(sample) for sample in samples]
        grouped_rollouts = [
            rollout_prompt(
                remote_url,
                rollout_model,
                prompt,
                group_size=group_size,
            )
            for prompt in prompts
        ]
        batch = build_grpo_batch(samples, prompts, grouped_rollouts)
        train_stats = dict(actor_step(batch) or {})
        sync_stats = sync_weights()
        yield {
            "step": step,
            "mean_reward": sum(item["reward"] for item in batch) / len(batch),
            "mean_advantage": sum(item["advantage"] for item in batch) / len(batch),
            "train": train_stats,
            "sync_requires_staging": bool(sync_stats["prepare"]["requires_staging"]),
            "sync_fallback_bytes": int(sync_stats["prepare"]["fallback_bytes"]),
        }
