"""Helpers for paper-aligned residual-mode benchmark comparisons.

These helpers operate on training traces recorded during real training runs.
They intentionally do not run training themselves; they only decide which
checkpoints to compare and how to answer questions such as:

- what loss did a mode reach at a given training budget?
- what is the latest checkpoint available within a time budget?
- when did a mode first match or beat a target loss?
"""

from __future__ import annotations

from collections.abc import Sequence

from .runtime import TrainingTracePoint


def resolve_checkpoint_steps(
    total_steps: int,
    *,
    checkpoint_every: int,
    explicit_steps: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Return sorted run steps to capture during a benchmarked training run."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")

    normalized_steps = {step for step in range(checkpoint_every, total_steps + 1, checkpoint_every)}
    normalized_steps.add(total_steps)

    if explicit_steps is not None:
        normalized_steps.update(
            step
            for step in explicit_steps
            if isinstance(step, int) and step > 0 and step <= total_steps
        )

    return tuple(sorted(normalized_steps))


def first_trace_meeting_loss(
    trace: Sequence[TrainingTracePoint],
    target_loss: float,
) -> TrainingTracePoint | None:
    """Return the first checkpoint whose loss is at or below the target."""
    for point in trace:
        if point.final_loss <= target_loss:
            return point
    return None


def latest_trace_within_elapsed(
    trace: Sequence[TrainingTracePoint],
    elapsed_budget: float,
) -> TrainingTracePoint | None:
    """Return the latest checkpoint still inside the elapsed-time budget."""
    candidates = [point for point in trace if point.elapsed <= elapsed_budget]
    if not candidates:
        return None
    return candidates[-1]
