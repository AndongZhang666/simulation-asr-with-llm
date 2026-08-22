import json
from pathlib import Path

from asr_ec.cli import main
from asr_ec.pipelines.prepare_data import run_prepare_data


def create_data_fixture(tmp_path: Path) -> Path:
    source_root = tmp_path / "source"
    chapter = source_root / "test-clean" / "1" / "2"
    chapter.mkdir(parents=True)
    (chapter / "1-2-0000.flac").write_bytes(b"fLaCfixture")
    (chapter / "1-2.trans.txt").write_text("1-2-0000 A TEST REFERENCE\n", encoding="utf-8")
    config = tmp_path / "data.yaml"
    config.write_text(
        "\n".join(
            [
                "dataset: librispeech",
                f"source_root: {source_root}",
                f"output_root: {tmp_path / 'manifests'}",
                f"runs_root: {tmp_path / 'runs'}",
                "normalizer: conservative-english@v1",
                "splits: [test-clean]",
                "expected_counts: {test-clean: 1}",
            ]
        ),
        encoding="utf-8",
    )
    return config


def test_prepare_data_dry_run_validates_without_writing_artifacts(tmp_path: Path) -> None:
    config = create_data_fixture(tmp_path)

    result = run_prepare_data(config, dry_run=True)

    assert result.dry_run is True
    assert result.artifacts[0].record_count == 1
    assert not (tmp_path / "manifests").exists()
    assert not (tmp_path / "runs").exists()


def test_prepare_data_writes_manifest_and_run_provenance(tmp_path: Path) -> None:
    config = create_data_fixture(tmp_path)

    result = run_prepare_data(config, dry_run=False)

    assert result.run_directory is not None
    assert (result.run_directory / "resolved_config.yaml").is_file()
    assert (result.run_directory / "run_manifest.json").is_file()
    assert result.artifacts[0].path.is_file()


def test_prepare_data_cli_emits_machine_readable_result(tmp_path: Path, capsys: object) -> None:
    config = create_data_fixture(tmp_path)

    assert main(["prepare-data", "--config", str(config), "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert json.loads(output)["dry_run"] is True
