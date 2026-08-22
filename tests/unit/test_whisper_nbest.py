import pytest

from asr_ec.asr.whisper_nbest import WhisperNBestError, rank_candidates


def test_rank_candidates_matches_whisper_default_length_normalization() -> None:
    candidates = rank_candidates(
        [(10, 11, 12), (20,), (30, 31)],
        [-3.0, -1.2, -2.4],
        length_penalty=None,
    )

    assert [candidate.rank for candidate in candidates] == [1, 2, 3]
    assert [candidate.token_ids for candidate in candidates] == [(10, 11, 12), (20,), (30, 31)]
    assert [candidate.ranking_score for candidate in candidates] == pytest.approx(
        [-1.0, -1.2, -1.2]
    )


def test_rank_candidates_uses_stable_tie_breaking() -> None:
    candidates = rank_candidates([(1,), (2,)], [-1.0, -1.0], length_penalty=None)

    assert [candidate.token_ids for candidate in candidates] == [(1,), (2,)]


def test_rank_candidates_uses_google_nmt_length_penalty() -> None:
    candidates = rank_candidates([(1,), (2, 3, 4)], [-1.0, -1.2], length_penalty=1.0)

    assert candidates[0].token_ids == (2, 3, 4)
    assert candidates[0].ranking_score == pytest.approx(-0.9)


@pytest.mark.parametrize(
    ("tokens", "scores", "penalty"),
    [([], [], None), ([(1,)], [-1.0, -2.0], None), ([()], [-1.0], None), ([(1,)], [-1.0], 1.1)],
)
def test_rank_candidates_rejects_invalid_finalization(
    tokens: list[tuple[int, ...]], scores: list[float], penalty: float | None
) -> None:
    with pytest.raises(WhisperNBestError):
        rank_candidates(tokens, scores, length_penalty=penalty)
