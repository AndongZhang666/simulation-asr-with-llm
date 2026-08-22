"""Command-line entry points for reproducible ASR-EC pipeline stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from asr_ec.evaluation.nbest import NBestEvaluationError
from asr_ec.pipelines.evaluate_nbest import run_evaluate_nbest
from asr_ec.pipelines.generate_nbest import NBestGenerationError, run_generate_nbest
from asr_ec.pipelines.prepare_data import DataPreparationError, run_prepare_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="asr-ec")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_data = subparsers.add_parser("prepare-data", help="build immutable dataset manifests")
    prepare_data.add_argument("--config", required=True, type=Path)
    prepare_data.add_argument("--dry-run", action="store_true")
    generate_nbest = subparsers.add_parser(
        "generate-nbest", help="generate immutable Whisper N-best ASR records"
    )
    generate_nbest.add_argument("--config", required=True, type=Path)
    generate_nbest.add_argument("--dry-run", action="store_true")
    evaluate_nbest = subparsers.add_parser("evaluate-nbest", help="score a stored N-best artifact")
    evaluate_nbest.add_argument("--artifact", required=True, type=Path)
    evaluate_nbest.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "prepare-data":
        try:
            data_result = run_prepare_data(arguments.config, dry_run=arguments.dry_run)
        except DataPreparationError as error:
            parser.error(str(error))
        print(json.dumps(data_result.to_dict(), ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    if arguments.command == "generate-nbest":
        try:
            nbest_result = run_generate_nbest(arguments.config, dry_run=arguments.dry_run)
        except NBestGenerationError as error:
            parser.error(str(error))
        print(json.dumps(nbest_result.to_dict(), ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    if arguments.command == "evaluate-nbest":
        try:
            evaluation_result = run_evaluate_nbest(arguments.artifact, dry_run=arguments.dry_run)
        except NBestEvaluationError as error:
            parser.error(str(error))
        print(json.dumps(evaluation_result.to_dict(), ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    parser.error(f"unsupported command: {arguments.command}")
    return 2
