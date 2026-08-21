"""User-facing command line interface for GWSamplegen-to-SNANA production."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catalog import prepare_dual_run_catalog, prepare_run_catalog
from config import load_profile
from migration import migrate_optical, prune_snana, validate_optical_coverage
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
    prepare.add_argument("--catalog", type=Path, help="Legacy single positive catalog")
    prepare.add_argument("--pos-catalog", type=Path)
    prepare.add_argument("--neg-catalog", type=Path)
    prepare.add_argument("--overwrite-prepared", action="store_true")

    submit = commands.add_parser("submit", help="Submit SNANA array and finalizer")
    submit.add_argument("profile")
    _add_submission_options(submit)

    run = commands.add_parser("run", help="Prepare a catalog, then submit it")
    run.add_argument("profile")
    run.add_argument("--catalog", type=Path, help="Legacy single positive catalog")
    run.add_argument("--pos-catalog", type=Path)
    run.add_argument("--neg-catalog", type=Path)
    run.add_argument("--overwrite-prepared", action="store_true")
    _add_submission_options(run)

    status = commands.add_parser("status", help="Summarize event status sidecars")
    status.add_argument("profile")

    compact = commands.add_parser("compact", help="Merge terminal artifacts into HDF5")
    compact.add_argument("profile")
    migrate = commands.add_parser(
        "migrate-optical", help="Archive existing SNANA outputs into v2 HDF5 shards"
    )
    migrate.add_argument("profile")
    migrate.add_argument("--batch-size", type=int, default=200)
    validate_optical = commands.add_parser(
        "validate-optical", help="Validate v2 optical coverage and checksums"
    )
    validate_optical.add_argument("profile")
    prune = commands.add_parser(
        "prune-snana", help="Dry-run or remove verified legacy SNANA directories"
    )
    prune.add_argument("profile")
    prune.add_argument("--execute", action="store_true")
    return parser


def _prepare(args: argparse.Namespace) -> dict:
    profile = load_profile(args.profile)
    legacy = args.catalog is not None
    dual = args.pos_catalog is not None or args.neg_catalog is not None
    if legacy == dual:
        raise ValueError(
            "provide either --catalog or both --pos-catalog and --neg-catalog"
        )
    if dual:
        if args.pos_catalog is None or args.neg_catalog is None:
            raise ValueError("dual preparation requires both catalog paths")
        if profile.negative_skymap_dir is None:
            raise ValueError("dual preparation requires paths.negative_skymap_dir")
        return prepare_dual_run_catalog(
            args.pos_catalog,
            args.neg_catalog,
            profile.run_dir,
            profile_name=profile.name,
            source=profile.source,
            split=profile.split,
            seed=profile.seed,
            positive_skymap_dir=profile.skymap_dir,
            negative_skymap_dir=profile.negative_skymap_dir,
            opsim_db=profile.opsim_db,
            mjd_min=profile.mjd_min,
            mjd_max=profile.mjd_max,
            profile=profile.as_manifest(),
            overwrite=args.overwrite_prepared,
        )
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
    elif args.command == "migrate-optical":
        result = migrate_optical(
            load_profile(args.profile), batch_size=int(args.batch_size)
        )
    elif args.command == "validate-optical":
        result = validate_optical_coverage(load_profile(args.profile))
    elif args.command == "prune-snana":
        result = prune_snana(load_profile(args.profile), execute=bool(args.execute))
    else:  # pragma: no cover - argparse enforces the command choices
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
