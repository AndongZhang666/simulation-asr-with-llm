# Reproducibility Gaps

## Historical Replication Is Currently Blocked

| Gap | Evidence | Effect | Planned handling |
| --- | --- | --- | --- |
| Author code and exact experiment drivers were not located. | `04_Open_Source_and_Reproducibility_Investigation.txt` Sec. 1 | Prevents attribution of this implementation to the authors. | Build an independent method reproduction with provenance. |
| Whisper revision and N-best/token-probability patch are not published. | PAPER Sec. V-A; investigation Sec. 5 | May change baseline, candidate diversity, and scores. | Pin public code; publish a small patch; require N=1 parity. |
| Exact Transducer checkpoint/config is unknown. | PAPER Sec. V-A; investigation Sec. 2 | A public ESPnet checkpoint may differ architecturally. | Compare every field and label mismatches. |
| ASR score normalization for interpolation is unspecified. | PAPER Eq. (2) | Could affect constrained decisions. | Use raw scores first; add clearly labeled length-normalized and within-list alternatives. |
| T5 separator tokenizer behavior, decoding settings, and several optimizer details are unspecified. | PAPER Secs. II-B and V-A | Exact model behavior cannot be reconstructed from the article alone. | Freeze literal `[SEP]` first; use development-only ablations. |
| Historical GPT snapshots are retired. | investigation Sec. 6 | Historical zero-shot numbers cannot be regenerated. | Cache all current-model calls and keep them separate from paper rows. |
| Internal MGB-3 and Linguaskill data are unavailable. | PAPER Sec. IV; investigation Sec. 3 | Full contamination table cannot be reproduced. | Run public-data-only diagnostic and identify the omission. |
| Lattice export and BPE-to-word equivalence details are incomplete. | PAPER Sec. III-D | Unverified lattice decoding would be scientifically unsafe. | Implement only after toy-graph exhaustive equivalence passes. |
| Whisper token-level final-beam probabilities are not exposed by the current adapter. | Inspected `openai-whisper==20250625` decoder | Word-confidence analysis cannot yet be reproduced. | Add a separately tested decoder instrumentation patch; preserve current cumulative-score artifacts. |

## Current Operational Gaps

- Network access is available; the official LibriSpeech `test-clean` archive and Whisper
  `small.en` checkpoint were downloaded and checksum-verified for bounded smoke tests.
- `uv`, `ruff`, `mypy`, Whisper, PyTorch, and FFmpeg are installed. Transformers, JiWER,
  SCTK, and a native PDF text extractor remain absent from the locked project environment.
- No API credentials or approved budget are available; zero-shot experiments remain disabled.
- There is no historical code commit to record before the first local implementation
  commit. Early run manifests will correctly identify the repository as newly initialized.

None of these gaps prevents Phase 1. They prohibit full-data/model execution until the
corresponding explicit entry gate is satisfied.