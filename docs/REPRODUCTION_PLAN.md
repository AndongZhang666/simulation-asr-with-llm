# Reproduction Plan

## Scope

This repository is an independent **Level B method reproduction** of Ma et al.,
"ASR Error Correction Using Large Language Models" (TASLP 2025), together with a
Level C engineering framework. It must not be presented as Level A historical
numerical replication unless author artifacts become available.

The initial target is the public LibriSpeech + Whisper `small.en` + N-best T5
pipeline. It will produce unconstrained, N-best-constrained, and closest results.
The ESPnet Transducer, lattice decoding, transfer, ensemble, and current-model
zero-shot work are subsequent gated phases.

## Current Status

- **M0 complete; M1 core complete**: typed records, schemas, configuration
   hashing, immutable artifacts, run provenance, normalization adapters,
   WER/CER/WERR, oracle, diagnostics, and strict quality tooling are implemented
   and tested. The required independent SCTK/JiWER fixture cross-check remains
   outstanding.
- **M2 smoke gate complete**: the official LibriSpeech `test-clean` archive was
   downloaded, structurally validated, and published as a 2,620-record immutable
   manifest.
- **M3 smoke path partially complete**: `openai-whisper==20250625`, checkpoint SHA-256
   `f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872`, and
   FFmpeg 9.0.1 are installed locally. The N=1 adapter exactly matches stock
   decoding on one public utterance. A two-utterance 10-best artifact and its
   baseline/oracle/diversity evaluation are published. Token-level probability
   tracing remains an explicit M3 exit-gate gap.
- **M4 foundations complete**: literal `[SEP]` serialization, immutable EC pairs,
   padding-safe teacher-forced sequence scoring, and unconstrained/constrained/
   closest strategy logic are tested. T5-base dependency installation, full training
   pairs, model training, and checkpoint-resume verification have not started.

## Source Integrity

- The five individual implementation documents were verified against
  `docs/CHECKSUMS.json` on 2026-08-21.
- The 13-page source paper was read with macOS PDFKit because `pdftotext` is not
  installed in this environment.
- The individual source documents take precedence over
  `05_Combined_All_in_One_Agent_Context.txt`, which is intentionally not used as
  an independent authority.

## Milestones

1. **Phase 0 / M0**: Create the installable package, typed contracts, immutable
   artifact tracking, configuration fingerprints, schemas, and CI-quality test
   commands.
2. **Phase 1 / M1**: Implement named text normalizers, edit alignment, corpus
   WER/CER, WERR, oracle selection, and correction diagnostics. Cross-check a
   fixed fixture with an independent scorer before ASR decoding.
3. **Phase 2 / M2**: Build deterministic LibriSpeech manifests and CI fixtures.
   Verify counts, overlap, references, audio paths, and hashes.
4. **Phase 3 / M3**: Patch a pinned official Whisper decoder to retain finalized
   N-best beams. Prove N=1 parity before baseline or oracle reporting.
5. **Phase 4 / M5**: Build immutable N-best-to-reference pairs and train T5-base
   using the paper settings: AdamW, learning rate `5e-5`, effective batch size 32,
   dropout 0.1, and three epochs.
6. **Phase 5 / M6**: Implement, test, and compare unconstrained, constrained, and
   closest decoding. Select lambda only on validation data over `0.00..1.00` in
   steps of `0.05`.
7. **Later phases**: Add transfer, N-best order and size ablations, model
   combinations, current-model zero-shot reproduction, the closest public ESPnet
   Transducer, and validated lattice search.

## Resource Assessment

Observed on 2026-08-21:

| Resource | Observation | Consequence |
| --- | --- | --- |
| Python | 3.10.17 | Package minimum is Python 3.10. |
| Compute | Apple M3 Pro with Metal support; 18 GB RAM | Phase 1 is CPU-only. Model training must start with small smoke runs and measure MPS memory. |
| Storage | 126 GiB available | Do not download full LibriSpeech plus duplicate checkpoints/artifacts until a storage budget is recorded. |
| Installed packages | Pydantic, PyYAML, pytest, PyTorch | Begin with standard-library core code; install model/tooling dependencies in a locked environment before later phases. |
| Repository | New Git repository | Official runs will record commit and dirty-tree status after the initial commit exists. |
| Network and APIs | Not verified; no approved API budget | Do not download models/datasets or invoke paid providers until the relevant gate and budget are recorded. |

Expected later-phase costs are estimates, not measurements: LibriSpeech 960h audio
and manifests require tens of GiB; a T5-base checkpoint is approximately 1 GiB plus
optimizer/checkpoint storage; full Whisper decoding and T5 fine-tuning should be
treated as multi-hour to multi-day MPS jobs. Each expensive command will provide a
fresh estimate from its resolved configuration before execution.

## Immediate Acceptance Criteria

Before ASR decoding, the repository must demonstrate all of the following:

- records validate and serialize without losing raw versus normalized text;
- configuration hashes are canonical and deterministic;
- artifacts are content-addressed and never overwritten;
- assumptions use an allowed evidence label;
- corpus WER and CER use summed edits, not mean utterance error rates;
- oracle selection has documented deterministic tie-breaking;
- unit tests cover hand-calculated edit cases and invalid contracts.

The pre-ASR gate and its bounded smoke continuation are satisfied. Full-corpus decoding
remains gated on a specific runtime/cost estimate and a clean published N=1 parity run.