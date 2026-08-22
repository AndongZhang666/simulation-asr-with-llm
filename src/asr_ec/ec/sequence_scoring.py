"""Padding-safe teacher-forced sequence log-probability aggregation for T5 candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class SequenceScoringError(ValueError):
    """Raised when model logits and candidate labels are incompatible."""


@dataclass(frozen=True, slots=True)
class SequenceScore:
    """A summed teacher-forced score and its documented non-padding token count."""

    log_probability_sum: float
    token_count: int
    mean_log_probability: float | None
    includes_eos: bool


def sequence_logprob_from_logits(
    logits: Any,
    labels: Any,
    *,
    ignore_index: int = -100,
    includes_eos: bool = True,
) -> tuple[SequenceScore, ...]:
    """Gather per-label log probabilities, excluding padding labels from sum and length.

    The caller controls EOS through the supplied labels and records that policy using
    ``includes_eos``. This function intentionally does not infer a model tokenizer.
    """

    import torch

    if logits.ndim != 3 or labels.ndim != 2:
        raise SequenceScoringError("logits must be [batch, time, vocab] and labels [batch, time]")
    if logits.shape[:2] != labels.shape:
        raise SequenceScoringError("logit batch/time dimensions must match labels")
    if labels.numel() == 0:
        raise SequenceScoringError("labels must not be empty")
    valid_mask = labels != ignore_index
    safe_labels = labels.masked_fill(~valid_mask, 0)
    if safe_labels.min().item() < 0 or safe_labels.max().item() >= logits.shape[-1]:
        raise SequenceScoringError("non-padding labels must be valid vocabulary IDs")
    token_logprobs = (
        torch.log_softmax(logits.float(), dim=-1)
        .gather(dim=-1, index=safe_labels.unsqueeze(-1))
        .squeeze(-1)
    )
    token_logprobs = token_logprobs.masked_fill(~valid_mask, 0)
    sums = token_logprobs.sum(dim=-1)
    counts = valid_mask.sum(dim=-1)
    return tuple(
        SequenceScore(
            log_probability_sum=float(score_sum.item()),
            token_count=int(count.item()),
            mean_log_probability=(float(score_sum.item() / count.item()) if count.item() else None),
            includes_eos=includes_eos,
        )
        for score_sum, count in zip(sums, counts, strict=True)
    )
