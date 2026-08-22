from pathlib import Path

import pytest

from asr_ec.tracking import (
    ArtifactStore,
    Assumption,
    AssumptionsRegistry,
    EvidenceLabel,
    canonical_json,
    config_sha256,
    create_run_manifest,
    run_id,
)
from asr_ec.tracking.assumptions import AssumptionsError
from asr_ec.tracking.config_hash import ConfigurationError
from asr_ec.tracking.run_manifest import RunManifestError


def test_config_hash_is_independent_of_mapping_order() -> None:
    first = {"experiment": {"seed": 13, "name": "smoke"}, "n": 10}
    second = {"n": 10, "experiment": {"name": "smoke", "seed": 13}}

    assert canonical_json(first) == canonical_json(second)
    assert config_sha256(first) == config_sha256(second)
    assert run_id("metrics", first) == run_id("metrics", second)


def test_config_hash_changes_for_meaningful_value() -> None:
    first = {"n": 5}
    second = {"n": 10}

    assert config_sha256(first) != config_sha256(second)


def test_config_rejects_nonfinite_values() -> None:
    with pytest.raises(ConfigurationError, match="NaN"):
        canonical_json({"learning_rate": float("nan")})


def test_artifact_store_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    first = store.put_bytes(b'{"utt_id":"one"}\n', suffix=".jsonl")
    second = store.put_bytes(b'{"utt_id":"one"}\n', suffix=".jsonl")

    assert first == second
    assert first.path.read_bytes() == b'{"utt_id":"one"}\n'
    assert first.path.name.startswith(first.sha256)


def test_assumptions_registry_is_append_only(tmp_path: Path) -> None:
    registry = AssumptionsRegistry()
    registry.add(
        Assumption(
            assumption_id="A1",
            choice="literal [SEP]",
            reason="paper-visible serialization",
            evidence_label=EvidenceLabel.INFERENCE,
        )
    )
    destination = tmp_path / "assumptions.json"
    registry.write_once(destination)

    with pytest.raises(AssumptionsError, match="overwrite"):
        registry.write_once(destination)
    with pytest.raises(AssumptionsError, match="duplicate"):
        registry.add(
            Assumption(
                assumption_id="A1",
                choice="other",
                reason="test",
                evidence_label=EvidenceLabel.OPEN_GAP,
            )
        )


def test_run_manifest_records_config_before_execution(tmp_path: Path) -> None:
    config = {"experiment": {"name": "metric-fixture", "seed": 13}}

    run_directory, manifest = create_run_manifest(
        tmp_path / "runs",
        prefix="metric-fixture",
        resolved_config=config,
        command=("asr-ec", "evaluate-nbest", "--dry-run"),
    )

    assert (run_directory / "resolved_config.yaml").is_file()
    assert (run_directory / "run_manifest.json").is_file()
    assert manifest.config_sha256 == config_sha256(config)
    with pytest.raises(RunManifestError, match="already exists"):
        create_run_manifest(
            tmp_path / "runs",
            prefix="metric-fixture",
            resolved_config=config,
            command=("asr-ec", "evaluate-nbest", "--dry-run"),
        )
