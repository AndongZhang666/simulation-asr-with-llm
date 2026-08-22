# ASR Error Correction With LLMs

This repository is an independent Level B method reproduction of Ma et al. (2025),
with a reusable Level C research framework. It is not the authors' source code and does
not claim historical numerical replication.

The current milestone implements typed, immutable records and will establish trustworthy
normalization and corpus metrics before any dataset download, ASR decoding, model
training, or paid API use.

See [docs/REPRODUCTION_PLAN.md](docs/REPRODUCTION_PLAN.md) for the scoped plan and
[docs/REPRODUCIBILITY_GAPS.md](docs/REPRODUCIBILITY_GAPS.md) for explicit limitations.

## Development

Use the committed `uv.lock` to create the development environment and run the Phase 1
quality checks:

```bash
uv sync --extra dev
make check
```
