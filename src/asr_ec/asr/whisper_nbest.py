"""Expose finalized official-Whisper beam candidates without modifying top-1 decoding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


class WhisperNBestError(ValueError):
    """Raised when finalized Whisper candidates cannot be represented faithfully."""


@dataclass(frozen=True, slots=True)
class NBestCandidate:
    """One finalized text candidate ranked with Whisper's installed ranker semantics."""

    rank: int
    text: str
    token_ids: tuple[int, ...]
    sequence_logscore: float
    ranking_score: float
    token_logprobs: tuple[float, ...] = ()
    token_logprobs_available: bool = False

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise WhisperNBestError("candidate rank must be positive")
        if not math.isfinite(self.sequence_logscore) or not math.isfinite(self.ranking_score):
            raise WhisperNBestError("candidate scores must be finite")
        if self.token_logprobs and len(self.token_ids) != len(self.token_logprobs):
            raise WhisperNBestError(
                "token IDs and token log probabilities must have matching lengths"
            )
        if self.token_logprobs_available != bool(self.token_logprobs):
            raise WhisperNBestError("token log-probability availability must match stored values")


@dataclass(frozen=True, slots=True)
class NBestDecodingResult:
    """All finalized candidates for one input audio item."""

    language: str
    candidates: tuple[NBestCandidate, ...]
    no_speech_prob: float

    def __post_init__(self) -> None:
        if not self.candidates:
            raise WhisperNBestError("a decoding result must contain at least one candidate")
        expected_ranks = tuple(range(1, len(self.candidates) + 1))
        if tuple(candidate.rank for candidate in self.candidates) != expected_ranks:
            raise WhisperNBestError("candidate ranks must be contiguous and score-ordered")


def rank_candidates(
    token_ids: Sequence[Sequence[int]],
    sequence_logprobs: Sequence[float],
    *,
    length_penalty: float | None,
) -> tuple[NBestCandidate, ...]:
    """Rank candidates with the exact formula in Whisper's MaximumLikelihoodRanker.

    When ``length_penalty`` is ``None``, Whisper divides the cumulative score by the
    generated-token count. Otherwise it uses the Google NMT length penalty. Python's
    stable sort preserves the upstream first-candidate behavior for equal scores.
    """

    if len(token_ids) != len(sequence_logprobs) or not token_ids:
        raise WhisperNBestError(
            "candidate tokens and scores must be non-empty and have equal lengths"
        )
    if length_penalty is not None and not 0 <= length_penalty <= 1:
        raise WhisperNBestError("length_penalty must be within [0, 1]")

    scored_candidates: list[tuple[float, int, tuple[int, ...], float]] = []
    for original_index, (tokens, sequence_logprob) in enumerate(
        zip(token_ids, sequence_logprobs, strict=True)
    ):
        token_tuple = tuple(tokens)
        if not token_tuple:
            raise WhisperNBestError("finalized candidate tokens must not be empty")
        if not math.isfinite(sequence_logprob):
            raise WhisperNBestError("sequence log probabilities must be finite")
        if length_penalty is None:
            penalty = len(token_tuple)
        else:
            penalty = ((5 + len(token_tuple)) / 6) ** length_penalty
        scored_candidates.append(
            (sequence_logprob / penalty, original_index, token_tuple, sequence_logprob)
        )

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    return tuple(
        NBestCandidate(
            rank=rank,
            text="",
            token_ids=tokens,
            sequence_logscore=sequence_logprob,
            ranking_score=ranking_score,
        )
        for rank, (ranking_score, _, tokens, sequence_logprob) in enumerate(
            scored_candidates, start=1
        )
    )


def decode_nbest(model: Any, mel: Any, options: Any) -> tuple[NBestDecodingResult, ...]:
    """Return every finalized beam from the installed official Whisper decoder.

    Imports are intentionally local so the metrics and data-manifest package remains
    usable without the optional ASR extra. This mirrors ``DecodingTask.run`` through
    ``decoder.finalize`` and replaces only the final one-best rank reduction.
    """

    if getattr(options, "beam_size", None) is None:
        raise WhisperNBestError("N-best extraction requires a deterministic Whisper beam_size")
    if getattr(options, "temperature", None) != 0:
        raise WhisperNBestError("N-best reproduction requires temperature=0")

    import torch
    from whisper.decoding import DecodingTask  # type: ignore[import-untyped]

    class FinalizedCandidatesTask(DecodingTask):  # type: ignore[misc]
        @torch.no_grad()
        def run_finalized(self, input_mel: Any) -> list[NBestDecodingResult]:
            self.decoder.reset()
            tokenizer = self.tokenizer
            n_audio = input_mel.shape[0]
            audio_features = self._get_audio_features(input_mel)
            tokens = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)
            languages, _ = self._detect_language(audio_features, tokens)
            if self.options.task == "lang_id":
                raise WhisperNBestError(
                    "language-identification mode does not produce text candidates"
                )

            tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)
            tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)
            no_speech_probs = no_speech_probs[:: self.n_group]
            tokens = tokens.reshape(n_audio, self.n_group, -1)
            sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)
            finalized_tokens, finalized_scores = self.decoder.finalize(tokens, sum_logprobs)

            results: list[NBestDecodingResult] = []
            for language, candidate_tensors, candidate_scores, no_speech_prob in zip(
                languages, finalized_tokens, finalized_scores, no_speech_probs, strict=True
            ):
                text_tokens = [
                    tensor[self.sample_begin : (tensor == tokenizer.eot).nonzero()[0, 0]].tolist()
                    for tensor in candidate_tensors
                ]
                ranked = rank_candidates(
                    text_tokens,
                    candidate_scores,
                    length_penalty=self.options.length_penalty,
                )
                decoded_candidates = tuple(
                    NBestCandidate(
                        rank=candidate.rank,
                        text=tokenizer.decode(list(candidate.token_ids)).strip(),
                        token_ids=candidate.token_ids,
                        sequence_logscore=candidate.sequence_logscore,
                        ranking_score=candidate.ranking_score,
                    )
                    for candidate in ranked
                )
                results.append(
                    NBestDecodingResult(
                        language=language,
                        candidates=decoded_candidates,
                        no_speech_prob=no_speech_prob,
                    )
                )
            return results

    task = FinalizedCandidatesTask(model, options)
    return tuple(task.run_finalized(mel))
