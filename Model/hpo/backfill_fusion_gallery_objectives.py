#!/usr/bin/env python3
"""Backfill FusionGallery objective metrics for interrupted HPO studies.

Older ALBEF_train.py builds saved best_ckpt_score for fusion_gallery_mrr but did
not save val_fusion_gallery_mrr or val_fusion_gallery_recall_at_1 in
trial_results.json. This utility recovers those fields from per-trial train.log
and can optionally update the Optuna SQLite study values.
"""

import argparse
import json
import math
import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

FUSION_GALLERY_RE = re.compile(
    r"FusionGallery:\s*R@1=([-+0-9.eE]+)\s+MRR=([-+0-9.eE]+)"
)
SUMMARY_NAMES = (
    "trial_results.json",
    "train_summary.json",
    "best_checkpoint_summary.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill missing FusionGallery HPO objective metrics."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="HPO output directory containing runtime config, results/, and optuna_study.db.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Runtime HPO config JSON. Defaults to the newest runtime_hpo_config_*.json in output-dir.",
    )
    parser.add_argument(
        "--storage-db",
        default=None,
        help="Optuna SQLite DB path. Defaults to output-dir/optuna_study.db.",
    )
    parser.add_argument(
        "--trials",
        default=None,
        help="Comma-separated trial numbers or ranges, e.g. 0,1,3-5. Defaults to all trial dirs.",
    )
    parser.add_argument(
        "--apply-json",
        action="store_true",
        help="Write recovered fields back to trial summary JSON files.",
    )
    parser.add_argument(
        "--apply-db",
        action="store_true",
        help="Update Optuna SQLite trial value and user attrs. Stop HPO jobs before using this.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite existing recovered FusionGallery fields.",
    )
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="Allow DB updates for RUNNING trials. Not recommended.",
    )
    return parser.parse_args()


def parse_trial_filter(text: Optional[str]) -> Optional[set]:
    if not text:
        return None
    out = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def newest_runtime_config(output_dir: Path) -> Path:
    configs = sorted(
        output_dir.glob("runtime_hpo_config_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not configs:
        raise FileNotFoundError(f"No runtime_hpo_config_*.json found in {output_dir}")
    return configs[0]


def load_weights(config_path: Path) -> Dict[str, float]:
    with config_path.open() as f:
        cfg = json.load(f)
    weights = cfg.get("objective_weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"{config_path} does not contain objective_weights")
    return {str(k): float(v) for k, v in weights.items()}


def trial_dirs(output_dir: Path, selected: Optional[set]) -> Iterable[Tuple[int, Path]]:
    results_dir = output_dir / "results"
    for path in sorted(results_dir.glob("trial_*"), key=lambda p: int(p.name.split("_", 1)[1])):
        try:
            number = int(path.name.split("_", 1)[1])
        except ValueError:
            continue
        if selected is not None and number not in selected:
            continue
        yield number, path


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def parse_fusion_metrics(log_path: Path) -> List[Tuple[float, float]]:
    metrics = []
    if not log_path.exists():
        return metrics
    with log_path.open(errors="replace") as f:
        for line in f:
            match = FUSION_GALLERY_RE.search(line)
            if match:
                metrics.append((float(match.group(1)), float(match.group(2))))
    return metrics


def finite_float(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def objective_score(results: dict, weights: Dict[str, float]) -> float:
    total = sum(weights.values())
    if total <= 0:
        return float("nan")
    score = 0.0
    for key, weight in weights.items():
        value = finite_float(results.get(key))
        if value is None:
            return float("nan")
        score += weight * value
    return score / total


def recover_trial(trial_dir: Path, weights: Dict[str, float], force: bool) -> Tuple[Optional[dict], str]:
    result_path = trial_dir / "checkpoints" / "ALBEF" / "trial_results.json"
    if not result_path.exists():
        return None, "missing trial_results.json"

    results = load_json(result_path)
    needs_mrr = force or "val_fusion_gallery_mrr" not in results
    needs_r1 = force or "val_fusion_gallery_recall_at_1" not in results
    if not needs_mrr and not needs_r1:
        score = objective_score(results, weights)
        return {
            "results": results,
            "score": score,
            "changed": False,
            "result_path": result_path,
        }, "already complete"

    best_epoch_1based = results.get("best_epoch_1based")
    if best_epoch_1based is None and "best_epoch" in results:
        best_epoch_1based = int(results["best_epoch"]) + 1
    if best_epoch_1based is None:
        return None, "missing best_epoch/best_epoch_1based"
    best_epoch_1based = int(best_epoch_1based)

    parsed = parse_fusion_metrics(trial_dir / "train.log")
    if not (1 <= best_epoch_1based <= len(parsed)):
        return None, f"cannot map best_epoch={best_epoch_1based} to parsed FusionGallery metrics ({len(parsed)})"

    fg_r1, fg_mrr_from_log = parsed[best_epoch_1based - 1]
    fg_mrr = fg_mrr_from_log
    if results.get("best_ckpt_metric") == "fusion_gallery_mrr":
        fg_mrr = finite_float(results.get("best_ckpt_score")) or fg_mrr_from_log

    updated = dict(results)
    if needs_mrr:
        updated["val_fusion_gallery_mrr"] = fg_mrr
    if needs_r1:
        updated["val_fusion_gallery_recall_at_1"] = fg_r1

    score = objective_score(updated, weights)
    if not math.isfinite(score):
        return None, "recovered metrics still do not satisfy objective weights"

    return {
        "results": updated,
        "score": score,
        "changed": True,
        "result_path": result_path,
        "best_epoch_1based": best_epoch_1based,
        "fg_mrr": fg_mrr,
        "fg_r1": fg_r1,
    }, "recovered"


def backup_path(path: Path, stamp: str) -> Path:
    return path.with_name(f"{path.name}.bak_backfill_{stamp}")


def write_trial_jsons(record: dict, stamp: str) -> None:
    summary_dir = record["result_path"].parent
    for name in SUMMARY_NAMES:
        path = summary_dir / name
        if not path.exists():
            continue
        shutil.copy2(path, backup_path(path, stamp))
        with path.open("w") as f:
            json.dump(record["results"], f, indent=2)
            f.write("\n")


def set_user_attr(conn: sqlite3.Connection, trial_id: int, key: str, value) -> None:
    value_json = json.dumps(value)
    row = conn.execute(
        "select trial_user_attribute_id from trial_user_attributes where trial_id=? and key=?",
        (trial_id, key),
    ).fetchone()
    if row:
        conn.execute(
            "update trial_user_attributes set value_json=? where trial_user_attribute_id=?",
            (value_json, row[0]),
        )
    else:
        conn.execute(
            "insert into trial_user_attributes(trial_id, key, value_json) values (?, ?, ?)",
            (trial_id, key, value_json),
        )


def update_db(db_path: Path, records: Dict[int, dict], include_running: bool, stamp: str) -> None:
    shutil.copy2(db_path, backup_path(db_path, stamp))
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute("begin immediate")
        for number, record in records.items():
            row = conn.execute(
                "select trial_id, state from trials where number=?",
                (number,),
            ).fetchone()
            if not row:
                print(f"trial {number}: DB skip missing trial row")
                continue
            trial_id, state = row
            if state != "COMPLETE" and not include_running:
                print(f"trial {number}: DB skip state={state}")
                continue

            value_row = conn.execute(
                "select trial_value_id from trial_values where trial_id=? and objective=0",
                (trial_id,),
            ).fetchone()
            if value_row:
                conn.execute(
                    "update trial_values set value=?, value_type='FINITE' where trial_value_id=?",
                    (float(record["score"]), value_row[0]),
                )
            else:
                conn.execute(
                    "insert into trial_values(trial_id, objective, value, value_type) values (?, 0, ?, 'FINITE')",
                    (trial_id, float(record["score"])),
                )

            set_user_attr(conn, trial_id, "objective_score", float(record["score"]))
            for key in (
                "val_fusion_gallery_mrr",
                "val_fusion_gallery_recall_at_1",
                "val_fusion_gallery_recall_at_5",
                "val_auprc",
                "val_auroc",
            ):
                if key in record["results"]:
                    set_user_attr(conn, trial_id, key, record["results"][key])
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve() if args.config else newest_runtime_config(output_dir)
    db_path = Path(args.storage_db).expanduser().resolve() if args.storage_db else output_dir / "optuna_study.db"
    selected = parse_trial_filter(args.trials)
    weights = load_weights(config_path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    print(f"output_dir: {output_dir}")
    print(f"config: {config_path}")
    print(f"weights: {weights}")
    print(f"mode: apply_json={int(args.apply_json)} apply_db={int(args.apply_db)}")

    recovered = {}
    for number, trial_dir in trial_dirs(output_dir, selected):
        record, reason = recover_trial(trial_dir, weights, args.force)
        if record is None:
            print(f"trial {number}: skip ({reason})")
            continue
        if record["changed"]:
            print(
                f"trial {number}: {reason} best_epoch={record['best_epoch_1based']} "
                f"FG_MRR={record['fg_mrr']:.6f} FG_R@1={record['fg_r1']:.6f} "
                f"objective={record['score']:.6f}"
            )
            recovered[number] = record
        else:
            print(f"trial {number}: {reason} objective={record['score']:.6f}")

    if not recovered:
        print("No trials need backfill.")
        return

    if args.apply_json:
        for record in recovered.values():
            write_trial_jsons(record, stamp)
        print(f"Wrote JSON summaries for {len(recovered)} trial(s). Backups suffix: .bak_backfill_{stamp}")

    if args.apply_db:
        if not db_path.exists():
            raise FileNotFoundError(db_path)
        update_db(db_path, recovered, args.include_running, stamp)
        print(f"Updated Optuna DB: {db_path}. Backup suffix: .bak_backfill_{stamp}")

    if not args.apply_json and not args.apply_db:
        print("Dry run only. Re-run with --apply-json and, after stopping HPO jobs, --apply-db to write changes.")


if __name__ == "__main__":
    main()
