"""Manifest-driven deterministic Whisper N-best generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from asr_ec.asr.whisper_nbest import decode_nbest
from asr_ec.asr.whisper_records import (
    NBestArtifact,
    build_utterance_record,
    write_nbest_artifact_once,
)
from asr_ec.data.manifests import DataManifestRecord
from asr_ec.data.normalization import TextNormalizer, WhisperEnglishNormalizer
from asr_ec.tracking.config_hash import config_sha256
from asr_ec.tracking.run_manifest import create_run_manifest


class NBestGenerationError(ValueError):
    """Raised when N-best generation would run with an incomplete or unsafe configuration."""


@dataclass(frozen=True, slots=True)
class NBestGenerationResult:
    artifact: NBestArtifact | None
    requested_records: int
    dry_run: bool
    run_directory: Path | None

    def to_dict(self) -> dict[str, object]:
        return {
            "dry_run": self.dry_run,
            "requested_records": self.requested_records,
            "run_directory": self.run_directory.as_posix() if self.run_directory else None,
            "artifact": (
                {
                    "sha256": self.artifact.sha256,
                    "path": self.artifact.path.as_posix(),
                    "record_count": self.artifact.record_count,
                }
                if self.artifact
                else None
            ),
        }


def run_generate_nbest(config_path: Path, *, dry_run: bool) -> NBestGenerationResult:
    """Validate a fixed Whisper configuration and optionally publish its N-best artifact."""

    config = _load_config(config_path)
    manifest_path = _required_path(config, "manifest_path")
    source_root = _required_path(config, "source_root")
    model_path = _required_path(config, "model_path")
    output_root = _required_path(config, "output_root")
    records = _load_manifest(manifest_path)
    selected_records = _select_records(records, config.get("max_records"))
    _validate_config_and_inputs(config, selected_records, source_root, model_path)

    if dry_run:
        return NBestGenerationResult(
            artifact=None,
            requested_records=len(selected_records),
            dry_run=True,
            run_directory=None,
        )

    runs_root = _required_path(config, "runs_root")
    run_directory, _ = create_run_manifest(
        runs_root,
        prefix="generate-nbest",
        resolved_config=config,
        command=("asr-ec", "generate-nbest", "--config", str(config_path)),
        input_artifact_hashes={"manifest": _file_sha256(manifest_path)},
        retry_on_collision=True,
    )
    decoded_records = _decode_records(config, selected_records, source_root)
    artifact = write_nbest_artifact_once(decoded_records, output_root=output_root)
    (run_directory / "output_artifacts.json").write_text(
        json.dumps({"nbest": artifact.sha256}, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return NBestGenerationResult(
        artifact=artifact,
        requested_records=len(selected_records),
        dry_run=False,
        run_directory=run_directory,
    )


def _load_config(config_path: Path) -> Mapping[str, Any]:
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise NBestGenerationError(f"could not read configuration: {config_path}") from error
    if not isinstance(config, dict):
        raise NBestGenerationError("N-best configuration must be a YAML mapping")
    return config


def _load_manifest(path: Path) -> tuple[DataManifestRecord, ...]:
    if not path.is_file():
        raise NBestGenerationError(f"manifest does not exist: {path}")
    records = tuple(
        DataManifestRecord.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not records:
        raise NBestGenerationError("manifest contains no records")
    return records


def _select_records(
    records: Sequence[DataManifestRecord], max_records: object
) -> tuple[DataManifestRecord, ...]:
    if max_records is None:
        return tuple(records)
    if not isinstance(max_records, int) or max_records < 1:
        raise NBestGenerationError("max_records must be a positive integer when provided")
    return tuple(records[:max_records])


def _validate_config_and_inputs(
    config: Mapping[str, Any],
    records: Sequence[DataManifestRecord],
    source_root: Path,
    model_path: Path,
) -> None:
    expected_values = {
        "backend": "whisper",
        "model": "small.en",
        "language": "en",
        "task": "transcribe",
        "score_type": "sum_logprob",
        "stochastic_fallback": False,
        "temperature": 0,
        "condition_on_previous_text": False,
    }
    for key, expected in expected_values.items():
        if config.get(key) != expected:
            raise NBestGenerationError(
                f"{key} must be {expected!r} for the deterministic reproduction"
            )
    nbest = config.get("nbest")
    if not isinstance(nbest, int) or nbest < 1:
        raise NBestGenerationError("nbest must be a positive integer")
    if config.get("beam_size") != nbest:
        raise NBestGenerationError(
            "beam_size must equal nbest to retain the requested finalized beams"
        )
    if not model_path.is_file():
        raise NBestGenerationError(f"Whisper model checkpoint does not exist: {model_path}")
    expected_model_sha256 = config.get("model_sha256")
    if (
        not isinstance(expected_model_sha256, str)
        or _file_sha256(model_path) != expected_model_sha256
    ):
        raise NBestGenerationError("Whisper model checkpoint hash does not match model_sha256")
    for record in records:
        audio_path = source_root / record.audio_path
        if not audio_path.is_file():
            raise NBestGenerationError(f"manifest audio file does not exist: {audio_path}")


def _decode_records(
    config: Mapping[str, Any], records: Sequence[DataManifestRecord], source_root: Path
) -> tuple[Any, ...]:
    import torch
    import whisper  # type: ignore[import-untyped]
    from whisper.decoding import DecodingOptions  # type: ignore[import-untyped]

    device = config.get("device", "cpu")
    if device not in {"cpu", "mps"}:
        raise NBestGenerationError("device must be cpu or mps")
    if device == "mps" and not torch.backends.mps.is_available():
        raise NBestGenerationError("MPS was requested but is unavailable")
    model = whisper.load_model(
        "small.en", download_root=str(Path(config["model_path"]).parent), device=device
    )
    options = DecodingOptions(
        language="en",
        task="transcribe",
        beam_size=config["beam_size"],
        temperature=0,
        fp16=False,
        without_timestamps=True,
    )
    normalizer: TextNormalizer = WhisperEnglishNormalizer(config["package_version"])
    checkpoint = f"small.en@sha256:{config['model_sha256']}"
    code_revision = f"openai-whisper@{config['package_version']}"
    decode_config_sha256 = config_sha256(
        {
            key: config[key]
            for key in (
                "beam_size",
                "condition_on_previous_text",
                "language",
                "nbest",
                "score_type",
                "stochastic_fallback",
                "task",
                "temperature",
            )
        }
    )
    lock_hash = _file_sha256(Path("uv.lock")) if Path("uv.lock").is_file() else None
    generated = []
    for record in records:
        audio = whisper.pad_or_trim(whisper.load_audio(str(source_root / record.audio_path)))
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        decoded = decode_nbest(model, mel.unsqueeze(0), options)[0]
        if len(decoded.candidates) != config["nbest"]:
            raise NBestGenerationError(
                "Whisper returned "
                f"{len(decoded.candidates)} candidates for {record.utt_id}, "
                f"expected {config['nbest']}"
            )
        generated.append(
            build_utterance_record(
                record,
                decoded,
                normalizer=normalizer,
                checkpoint=checkpoint,
                code_revision=code_revision,
                decode_config_sha256=decode_config_sha256,
                length_penalty=None,
                software_lock_sha256=lock_hash,
            )
        )
    return tuple(generated)


def _required_path(config: Mapping[str, Any], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise NBestGenerationError(f"{key} must be a non-empty path string")
    return Path(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
