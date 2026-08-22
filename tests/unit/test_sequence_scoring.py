import math

import pytest
import torch

from asr_ec.ec.sequence_scoring import SequenceScoringError, sequence_logprob_from_logits


def test_teacher_forced_sequence_score_excludes_padding_from_sum_and_length() -> None:
    logits = torch.tensor([[[math.log(0.1), math.log(0.7), math.log(0.2)], [0.0, 0.0, 0.0]]])
    labels = torch.tensor([[1, -100]])

    score = sequence_logprob_from_logits(logits, labels, includes_eos=True)[0]

    assert score.log_probability_sum == pytest.approx(math.log(0.7))
    assert score.token_count == 1
    assert score.mean_log_probability == pytest.approx(math.log(0.7))
    assert score.includes_eos is True


def test_teacher_forced_sequence_score_reports_multiple_batch_items() -> None:
    logits = torch.zeros((2, 1, 2))
    labels = torch.tensor([[0], [1]])

    scores = sequence_logprob_from_logits(logits, labels, includes_eos=False)

    assert [score.log_probability_sum for score in scores] == pytest.approx(
        [-math.log(2), -math.log(2)]
    )
    assert [score.token_count for score in scores] == [1, 1]


def test_teacher_forced_sequence_score_rejects_incompatible_shapes() -> None:
    with pytest.raises(SequenceScoringError, match="match labels"):
        sequence_logprob_from_logits(torch.zeros((1, 2, 3)), torch.zeros((1, 1), dtype=torch.long))
