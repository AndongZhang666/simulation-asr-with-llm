# Source-to-Code Map

This map links each major reported method to its planned implementation, test,
configuration, and immutable output. `PAPER` references the 2025 article; other
labels appear in the assumptions registry.

| Paper component | Source | Implementation | Test | Configuration | Output |
| --- | --- | --- | --- | --- | --- |
| Versioned ASR N-best record | PAPER Sec. II-A; spec Sec. 2 | `asr_ec.contracts.records` | `tests/unit/test_records.py` | ASR configs | `processed/nbest/<hash>/records.jsonl` |
| Ordered `[SEP]` serialization | PAPER Sec. II-B, Fig. 1 | `asr_ec.ec.input_serialization` | `tests/unit/test_serialization.py` | `configs/ec/t5_base.yaml` | EC pair JSONL |
| T5-base optimization | PAPER Sec. V-A | `asr_ec.ec.t5_training` | `tests/integration/test_t5_tiny_overfit.py` | `configs/ec/t5_base.yaml` | checkpoint + training metrics |
| Corpus WER/CER/WERR | PAPER Sec. V; spec Sec. 8 | `asr_ec.evaluation.metrics` | `tests/unit/test_metrics.py` | evaluation section of experiment configs | `metrics.json` |
| N-best oracle WER | PAPER Table II; spec Sec. 8 | `asr_ec.evaluation.oracle` | `tests/unit/test_oracle.py` | `configs/asr/*.yaml` | `oracle_metrics.json` |
| Whisper finalized N-best extraction | PAPER Sec. V-A; spec Sec. 6.2 | `asr_ec.asr.whisper_nbest` and a reviewable patch | `tests/integration/test_whisper_parity.py` | `configs/asr/whisper_small_en_nbest10.yaml` | N-best artifact + patch hash |
| Unconstrained decoding | PAPER Eq. (1) | `asr_ec.decoding.unconstrained` | `tests/unit/test_unconstrained.py` | `configs/decoding/unconstrained.yaml` | predictions JSONL |
| N-best constrained decoding | PAPER Eq. (2), Sec. III-B | `asr_ec.decoding.nbest_constrained` | `tests/unit/test_interpolation.py` | `configs/decoding/nbest_constrained.yaml` | lambda curve + predictions |
| Closest projection | PAPER Eq. (3), Sec. III-C | `asr_ec.decoding.closest` | `tests/unit/test_closest.py` | `configs/decoding/closest.yaml` | predictions JSONL |
| Lattice constrained decoding | PAPER Eq. (4), Algorithm 1 | `asr_ec.decoding.lattice` | `tests/unit/test_lattice_toy.py` | `configs/decoding/lattice.yaml` | lattice + validated predictions |
| Current-model zero-shot EC | PAPER Sec. II-C; prompts in Fig. 2 | `asr_ec.providers` and `asr_ec.decoding.llm_zero_shot` | `tests/unit/test_llm_parser.py` | `configs/decoding/zero_shot.yaml` | cached exchanges + predictions |
| Data-contamination diagnostic | PAPER Sec. IV | `asr_ec.evaluation.contamination` | `tests/unit/test_contamination.py` | `configs/experiments/contamination.yaml` | immutable quiz exchanges |
| Provenance and run records | architecture Secs. 1, 5, 6 | `asr_ec.tracking` | `tests/unit/test_tracking.py` | all commands | `runs/<run-id>/` manifest bundle |

The first implementation slice covers contracts, tracking, named normalization, and
metrics only. ASR, model, and provider modules remain deliberately absent until their
entry gates pass.