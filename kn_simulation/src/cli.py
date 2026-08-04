"""User-facing command line interface for GWSamplegen-to-SNANA production."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catalog import prepare_run_catalog
from config import load_profile
from scheduler import compact_profile, status_report, submit_profile


def _add_submission_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-concurrency", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kn-sim",
        description="Prepare GWSamplegen catalogs and run Rubin/SNANA simulations.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Build a validated kn_catalog.csv")
    prepare.add_argument("profile")
    prepare.add_argument("--catalog", required=True, type=Path)
    prepare.add_argument("--overwrite-prepared", action="store_true")

    submit = commands.add_parser("submit", help="Submit SNANA array and finalizer")
    submit.add_argument("profile")
    _add_submission_options(submit)

    run = commands.add_parser("run", help="Prepare a catalog, then submit it")
    run.add_argument("profile")
    run.add_argument("--catalog", required=True, type=Path)
    run.add_argument("--overwrite-prepared", action="store_true")
    _add_submission_options(run)

    status = commands.add_parser("status", help="Summarize event status sidecars")
    status.add_argument("profile")

    compact = commands.add_parser("compact", help="Merge terminal artifacts into HDF5")
    compact.add_argument("profile")
    return parser


def _prepare(args: argparse.Namespace) -> dict:
    profile = load_profile(args.profile)
    return prepare_run_catalog(
        args.catalog,
        profile.run_dir,
        profile_name=profile.name,
        source=profile.source,
        split=profile.split,
        seed=profile.seed,
        skymap_dir=profile.skymap_dir,
        opsim_db=profile.opsim_db,
        mjd_min=profile.mjd_min,
        mjd_max=profile.mjd_max,
        profile=profile.as_manifest(),
        overwrite=args.overwrite_prepared,
    )


def _submit(args: argparse.Namespace) -> dict:
    profile = load_profile(args.profile)
    return submit_profile(
        profile,
        args.profile,
        dry_run=args.dry_run,
        resume=args.resume,
        batch_size=args.batch_size,
        max_concurrency=args.max_concurrency,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = _prepare(args)
    elif args.command == "submit":
        result = _submit(args)
    elif args.command == "run":
        _prepare(args)
        result = _submit(args)
    elif args.command == "status":
        result = status_report(load_profile(args.profile))
    elif args.command == "compact":
        result = compact_profile(load_profile(args.profile))
    else:  # pragma: no cover - argparse enforces the command choices
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
