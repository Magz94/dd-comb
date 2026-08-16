"""Unified command-line interface for the ddcomb project.

This is a thin dispatcher over the Hydra entry-point scripts in ``scripts/``.
Every experiment can be launched either the historical way::

    python scripts/train_fusion.py mode=stage_d dataset=deepweeds_no_bg_mergeval

or through the unified CLI (identical behaviour, same Hydra overrides)::

    ddcomb train-fusion mode=stage_d dataset=deepweeds_no_bg_mergeval

The dispatcher execs the target script in-process with ``runpy`` and forwards
all remaining arguments to Hydra untouched, so the two invocation styles are
exactly equivalent. This keeps the SLURM jobs (which call the scripts directly)
working unchanged while giving a single clean public entry point.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

# Public command name -> entry-point script under ``scripts/``.
COMMANDS: dict[str, str] = {
    "train-fusion": "train_fusion.py",
    "train-fusion-to-json": "train_fusion_full_to_json.py",
    "train-single-backbone": "train_single_backbone.py",
    "sweep-kshot": "sweep_kshot.py",
    "sweep-kshot-baselines": "sweep_kshot_baselines.py",
    "sweep-kshot-ddcomb": "sweep_kshot_ddcobe.py",
    "sweep-single-backbone-kshot": "sweep_single_backbone_kshot.py",
    "run-modern-backbone": "run_modern_backbone_experiment.py",
    "run-modern-baselines": "run_modern_baselines.py",
    "run-deep-ensemble": "run_vanilla_deep_ensemble.py",
    "build-manifests": "build_kshot_manifests.py",
    "eval": "eval.py",
}


def _repo_root() -> Path:
    # src/ddcomb/cli.py -> parents[2] == repository root
    return Path(__file__).resolve().parents[2]


def _scripts_dir() -> Path:
    return _repo_root() / "scripts"


def _print_help() -> None:
    print("Usage: ddcomb <command> [hydra overrides ...]\n")
    print("Commands:")
    width = max(len(name) for name in COMMANDS)
    for name, script in COMMANDS.items():
        print(f"  {name.ljust(width)}  ->  scripts/{script}")
    print("\nExamples:")
    print("  ddcomb train-fusion mode=stage_d dataset=deepweeds_no_bg_mergeval")
    print("  ddcomb sweep-kshot dataset=plant_seedlings_v2_split")
    print("\nHydra overrides are passed through verbatim; --help on a command")
    print("shows that script's own configuration.")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help()
        return

    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        sys.stderr.write(f"ddcomb: unknown command '{command}'\n\n")
        _print_help()
        raise SystemExit(2)

    script = _scripts_dir() / COMMANDS[command]
    if not script.is_file():
        sys.stderr.write(f"ddcomb: entry-point script not found: {script}\n")
        raise SystemExit(1)

    # Hand off to the Hydra script exactly as if it were called directly.
    sys.argv = [str(script), *rest]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
