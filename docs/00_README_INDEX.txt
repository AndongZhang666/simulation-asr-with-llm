ASR ERROR CORRECTION USING LARGE LANGUAGE MODELS
REPRODUCTION DOCUMENTATION PACKAGE

Primary source
--------------
Rao Ma, Mengjie Qian, Mark Gales, and Kate Knill,
"ASR Error Correction Using Large Language Models," IEEE Transactions on
Audio, Speech and Language Processing, volume 33, 2025, pages 1389-1401.
DOI: 10.1109/TASLPRO.2025.3551083

Package purpose
---------------
This package converts the paper into implementation-oriented specifications for
an AI coding agent or a human research engineer. It is not a verbatim copy of
the article. It focuses on the information needed to rebuild, test, and report:

1. Supervised ASR error correction using N-best inputs and a fine-tuned T5 model.
2. Zero-shot ASR error correction using a generative LLM and N-best prompts.
3. Unconstrained, N-best-constrained, N-best-closest, and lattice-constrained decoding.
4. Cross-domain and cross-ASR evaluation.
5. Multi-ASR N-best combination and ROVER comparison.
6. N-best diversity analysis and a data-contamination diagnostic.

Evidence labels
---------------
Every important statement in these files is marked, where useful, with one of the
following labels:

[PAPER]           Directly supported by the target 2025 paper.
[REFERENCE-PAPER] Added from a paper explicitly cited by the target paper, most
                  importantly the earlier N-best T5 work by the same group.
[WEB-VERIFIED]    Verified against a public project, repository, model page, or
                  current service documentation during the investigation.
[INFERENCE]       A reasoned reconstruction needed to turn the paper into code.
[RECOMMENDATION]  An engineering choice intended to improve reproducibility.
[OPEN-GAP]        A detail that the paper does not specify sufficiently for exact
                  replication.

Files in this package
---------------------
00_README_INDEX.txt
    This index, reading order, reproduction boundaries, and quick-start plan.

01_Paper_to_Implementation_Reproduction_Spec.txt
    Detailed paper-to-code specification. It includes algorithms, data formats,
    prompts, experimental settings, target result tables, acceptance gates, and
    an explicit ambiguity ledger.

02_Codebase_Architecture_and_Module_Contracts.txt
    Proposed repository architecture, component interfaces, schemas, configuration
    rules, artifact management, testing strategy, and extension points.

03_AI_Agent_Rebuild_Runbook.txt
    A standalone instruction document for an AI coding agent. It divides the work
    into gated milestones and prevents the agent from silently inventing missing
    experimental details.

04_Open_Source_and_Reproducibility_Investigation.txt
    Investigation of exact and adjacent public code. It records what appears to be
    available, what is reusable, what is not the target implementation, and which
    historical dependencies block exact numerical reproduction.

05_Combined_All_in_One_Agent_Context.txt
    Concatenation of files 00-04 with section delimiters. Use this when the coding
    agent accepts one large context file more reliably than several attachments.

Recommended reading order
-------------------------
1. Read this file.
2. Give files 01, 02, and 03 together to the coding agent, or give it file 05.
3. Use file 04 before choosing dependencies or claiming an exact reproduction.
4. Use the paper PDF when resolving equations, diagrams, or table values.

Three levels of reproduction
----------------------------
Level A - Historical numerical replication
    Reproduce the same ASR hypotheses, model snapshots, API model versions,
    preprocessing, random seeds, and evaluation scripts. This is the strictest goal.
    It is currently blocked by missing author code/configuration, an unpublished
    Whisper N-best patch, uncertain ASR checkpoint identity, and retirement of the
    exact historical ChatGPT model snapshots.

Level B - Method reproduction
    Reimplement the described algorithms with public components, preserve the data
    flow and objectives, and target the paper's relative trends and approximately
    similar WER. This is the recommended main goal.

Level C - Engineering reconstruction
    Build a clean, reusable platform that supports the paper's methods plus modern
    replacements. This is useful for subsequent research even when exact historical
    numbers cannot be matched.

Recommended project scope
-------------------------
The most defensible first publication-quality reproduction is:

- English only.
- LibriSpeech test-clean and test-other first.
- Public Whisper small.en or a pinned equivalent revision.
- Ten unique beam candidates with raw scores and full provenance.
- T5-base supervised EC trained from the generated N-best lists.
- Unconstrained, N-best-constrained, and closest decoding.
- Oracle WER, WER, WERR, substitution/deletion/insertion breakdown, and N-best
  diversity diagnostics.
- Lattice-constrained decoding as a second-stage extension because it requires much
  deeper access to the ASR decoder.
- Zero-shot prompting as a method-compatible modern experiment, not an exact repeat
  of the paper's GPT-3.5/GPT-4 numbers.

Non-negotiable reproducibility rules
------------------------------------
- Never overwrite raw hypotheses or references after creation.
- Save raw and normalized text separately.
- Save every N-best candidate, its rank, token IDs when possible, and the exact score
  definition.
- Pin code revisions, model revisions, package versions, normalizer revisions, and
  dataset manifests.
- Determine baseline and oracle WER before training T5.
- Do not compare a reproduced EC WER against the paper unless the underlying ASR
  baseline and normalization are sufficiently close.
- Record every unspecified choice in an assumptions registry.
- Report method-compatible modern LLM results separately from historical paper results.

Quick-start execution order
---------------------------
1. Create repository and schemas.
2. Implement normalization and WER tests.
3. Prepare deterministic LibriSpeech manifests.
4. Generate and validate 10-best ASR artifacts.
5. Reproduce baseline and oracle WER.
6. Build and train N-best T5.
7. Add unconstrained, constrained, and closest decoding.
8. Run ablations on N and hypothesis order.
9. Add out-of-domain, cross-ASR, ensembling, lattice, and contamination experiments.
10. Produce a paper-table-compatible report plus a limitations section.

Source-code status in one paragraph
-----------------------------------
No official repository for the exact 2025 paper was located in the targeted search
performed for this package. Public implementations exist for the major building blocks
(Whisper, ESPnet, T5/Transformers, SCTK/ROVER, datasets, and several adjacent ASR EC
projects), but they should not be described as the authors' implementation. The exact
Whisper N-best/token-probability modification and the exact training/experiment driver
remain reproduction work.
