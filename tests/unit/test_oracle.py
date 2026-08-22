from asr_ec.contracts.records import Hypothesis, TextFields
from asr_ec.data.normalization import IdentityNormalizer
from asr_ec.evaluation.oracle import select_oracle


def make_hypothesis(rank: int, text: str) -> Hypothesis:
    return Hypothesis(
        rank=rank,
        text=TextFields(raw=text, normalized=text),
        source_system_id="fixture",
        source_original_rank=rank,
    )


def test_oracle_selects_lowest_edit_candidate() -> None:
    selection = select_oracle(
        "the correct answer",
        (make_hypothesis(1, "the wrong answer"), make_hypothesis(2, "the correct answer")),
        normalizer=IdentityNormalizer(),
        utt_id="fixture-1",
    )

    assert selection.selected_rank == 2
    assert selection.counts.errors == 0


def test_oracle_uses_rank_for_equal_edit_ties() -> None:
    selection = select_oracle(
        "a b",
        (make_hypothesis(1, "a c"), make_hypothesis(2, "a d")),
        normalizer=IdentityNormalizer(),
    )

    assert selection.selected_rank == 1
