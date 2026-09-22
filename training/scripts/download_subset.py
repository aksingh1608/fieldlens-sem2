#!/usr/bin/env python3
"""Build FieldLens subset from Agriculture-Vision-2021.tar.gz and write split.csv.

Extract uses ONE sequential pass over the .tar.gz (fast for gzip). Random
per-file seeks are avoided on purpose - they can take many hours.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fieldlens.constants import ANOMALY_CLASSES  # noqa: E402
from fieldlens.paths import CONFIG_DIR, REPO_ROOT as ROOT, repo_path  # noqa: E402
from fieldlens.profile import (  # noqa: E402
    add_profile_arg,
    load_profile,
    merge_data_with_profile,
)


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def field_id_from_name(name: str) -> str:
    stem = Path(name).stem
    if "_" not in stem:
        raise ValueError(f"unexpected tile name (no field id underscore): {name}")
    return stem.split("_", 1)[0]


def tile_index_cache_path(archive: Path) -> Path:
    return archive.with_suffix(archive.suffix + ".tile_ids.json")


def build_or_load_tile_index(archive: Path) -> dict[str, list[str]]:
    """Return {train|val|test: [tile_id, ...]} using a cache next to the archive."""
    cache = tile_index_cache_path(archive)
    if cache.is_file():
        data = json.loads(cache.read_text())
        print(f"Loaded tile id cache: {cache}")
        return data

    print(f"Building tile id cache from {archive} (one full scan, 15-40 min once)...")
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            # Agriculture-Vision-2021/{split}/images/rgb/{tile}.jpg
            parts = m.name.split("/")
            if len(parts) < 5:
                continue
            if parts[0] != "Agriculture-Vision-2021":
                continue
            split, kind, channel = parts[1], parts[2], parts[3]
            if split not in out or kind != "images" or channel != "rgb":
                continue
            if not m.name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            out[split].append(Path(m.name).stem)

    for k in out:
        out[k] = sorted(set(out[k]))
        print(f"  cached {k}: {len(out[k])} tiles")
    cache.write_text(json.dumps(out))
    print(f"Wrote {cache}")
    return out


def list_tiles_on_disk(extracted_root: Path, source_split: str) -> list[str]:
    rgb_dir = extracted_root / source_split / "images" / "rgb"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"missing {rgb_dir}; extract archive first or fix layout")
    return [
        p.stem
        for p in sorted(rgb_dir.glob("*"))
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]


def limit_tiles_per_field(
    tiles: list[str], max_per_field: int | None, seed: int
) -> list[str]:
    """Shuffle within each field and keep at most max_per_field tiles."""
    if max_per_field is None or max_per_field <= 0:
        return list(tiles)
    by_field: dict[str, list[str]] = defaultdict(list)
    for t in tiles:
        by_field[field_id_from_name(t)].append(t)
    rng = random.Random(seed)
    out: list[str] = []
    for fid in sorted(by_field.keys()):
        flist = list(by_field[fid])
        rng.shuffle(flist)
        out.extend(flist[: int(max_per_field)])
    return out


def assign_fields(
    tiles: list[str],
    n_first: int,
    n_second: int,
    seed: int,
    max_per_field_first: int | None = None,
    max_per_field_second: int | None = None,
) -> tuple[list[str], list[str]]:
    """Group by field, shuffle fields, fill first split then second from remaining fields.

    Optional per field caps are applied when a field is added to a split so more
    fields are needed to reach the same tile budgets.
    """
    by_field: dict[str, list[str]] = defaultdict(list)
    for t in tiles:
        by_field[field_id_from_name(t)].append(t)
    fields = list(by_field.keys())
    rng = random.Random(seed)
    rng.shuffle(fields)

    def take_from_field(fid: str, cap: int | None) -> list[str]:
        flist = list(by_field[fid])
        rng.shuffle(flist)
        if cap is None or cap <= 0:
            return flist
        return flist[: int(cap)]

    first: list[str] = []
    second: list[str] = []
    used: set[str] = set()
    for fid in fields:
        if len(first) >= n_first:
            break
        first.extend(take_from_field(fid, max_per_field_first))
        used.add(fid)
    for fid in fields:
        if fid in used:
            continue
        if n_second <= 0 or len(second) >= n_second:
            break
        second.extend(take_from_field(fid, max_per_field_second))
        used.add(fid)
    return first, second


def max_tiles_per_field_cfg(cfg: dict) -> dict[str, int | None]:
    raw = cfg.get("max_tiles_per_field")
    if raw is None:
        return {"train": None, "val": None, "test": None}
    if isinstance(raw, int):
        return {"train": raw, "val": raw, "test": raw}
    if not isinstance(raw, dict):
        raise ValueError("max_tiles_per_field must be an int or a mapping")
    return {
        "train": raw.get("train"),
        "val": raw.get("val"),
        "test": raw.get("test"),
    }


def apply_strict_tile_cap(
    splits: dict[str, list[str]], caps: dict[str, int]
) -> dict[str, list[str]]:
    """Hard-cut each split to at most caps[split] tiles (keeps field order; may partial-cut last field)."""
    out = {}
    for name, tiles in splits.items():
        cap = int(caps.get(name, len(tiles)))
        out[name] = tiles[:cap]
    return out


def assert_no_field_overlap(splits: dict[str, list[str]]) -> None:
    field_sets = {k: {field_id_from_name(t) for t in v} for k, v in splits.items()}
    keys = list(field_sets)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            overlap = field_sets[a] & field_sets[b]
            if overlap:
                raise RuntimeError(
                    f"FIELD OVERLAP between {a} and {b}: {sorted(overlap)[:20]}"
                )


def needed_name_set(split_rows: list[dict[str, str]]) -> set[str]:
    """All archive member basenames we might want (jpg and png variants for images)."""
    names: set[str] = set()
    for row in split_rows:
        tile_id = row["tile_id"]
        source_split = row["source_split"]
        base = f"Agriculture-Vision-2021/{source_split}"
        for path in (
            f"{base}/images/rgb/{tile_id}.jpg",
            f"{base}/images/rgb/{tile_id}.png",
            f"{base}/images/nir/{tile_id}.jpg",
            f"{base}/images/nir/{tile_id}.png",
            f"{base}/boundaries/{tile_id}.png",
            f"{base}/masks/{tile_id}.png",
        ):
            names.add(path)
        for c in ANOMALY_CLASSES:
            names.add(f"{base}/labels/{c}/{tile_id}.png")
    return names


def extract_subset(
    archive: Path,
    subset_root: Path,
    split_rows: list[dict[str, str]],
) -> None:
    """One sequential pass over the gzip tar; copy only needed members."""
    wanted = needed_name_set(split_rows)
    print(
        f"Sequential extract from {archive} "
        f"(want up to {len(wanted)} member names, {len(split_rows)} tiles)..."
    )
    written = 0
    seen = 0
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            seen += 1
            if seen % 100000 == 0:
                print(f"  scanned {seen} members, wrote {written} files...")
            if not member.isfile() or member.name not in wanted:
                continue
            rel = member.name[len("Agriculture-Vision-2021/") :]
            dest = subset_root / rel
            if dest.is_file() and dest.stat().st_size > 0:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                raise RuntimeError(f"could not extract {member.name}")
            dest.write_bytes(src.read())
            written += 1
    print(f"Extract finished: scanned={seen} wrote={written}")

    # Quick completeness check for rgb
    missing_rgb = []
    for row in split_rows:
        tid, src = row["tile_id"], row["source_split"]
        jpg = subset_root / src / "images" / "rgb" / f"{tid}.jpg"
        png = subset_root / src / "images" / "rgb" / f"{tid}.png"
        if not jpg.is_file() and not png.is_file():
            missing_rgb.append(tid)
    if missing_rgb:
        raise FileNotFoundError(
            f"missing rgb for {len(missing_rgb)} tiles after extract; "
            f"examples: {missing_rgb[:5]}"
        )


def write_split_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["tile_id", "source_split", "our_split", "field_id"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_DIR / "data.yaml"))
    add_profile_arg(parser)
    parser.add_argument(
        "--mode",
        choices=["plan", "extract", "smoke5"],
        default="plan",
        help="plan: write split.csv only; extract: copy subset; smoke5: 5-tile extract",
    )
    parser.add_argument(
        "--from-disk",
        action="store_true",
        help="List tiles from extracted_root instead of the archive/cache",
    )
    args = parser.parse_args()
    profile = load_profile(args.profile)
    cfg = merge_data_with_profile(profile)
    # Allow --config only for archive path overrides; profile supplies sizes/paths.
    if args.config:
        base = load_config(Path(args.config))
        for key in ("archive_path", "extracted_root", "bucket", "prefix", "archive_key"):
            if key in base:
                cfg[key] = base[key]
    print(f"profile={args.profile}")

    archive = repo_path(cfg["archive_path"]).resolve()
    extracted = repo_path(cfg["extracted_root"]).resolve()
    subset_root = repo_path(cfg["subset_root"]).resolve()
    split_csv = repo_path(cfg.get("split_csv", "data/fieldlens/split.csv")).resolve()

    def tiles_for(split: str) -> list[str]:
        if args.from_disk:
            return list_tiles_on_disk(extracted, split)
        if not archive.is_file():
            raise FileNotFoundError(
                f"Archive not found: {archive}. Download it first (see docs/data.md)."
            )
        index = build_or_load_tile_index(archive)
        return list(index[split])

    print("Listing official train tiles...")
    train_tiles = tiles_for("train")
    print(f"  official train tiles: {len(train_tiles)}")
    print("Listing official val tiles...")
    val_tiles = tiles_for("val")
    print(f"  official val tiles: {len(val_tiles)}")

    seed = int(cfg.get("seed", 42))
    caps = max_tiles_per_field_cfg(cfg)
    print(
        f"max_tiles_per_field train={caps['train']} val={caps['val']} test={caps['test']}"
    )

    our_train, our_val = assign_fields(
        train_tiles,
        cfg["n_train"],
        cfg["n_val"],
        seed,
        max_per_field_first=caps["train"],
        max_per_field_second=caps["val"],
    )
    our_test, _ = assign_fields(
        val_tiles,
        cfg["n_test"],
        0,
        seed,
        max_per_field_first=caps["test"],
        max_per_field_second=None,
    )
    splits = {"train": our_train, "val": our_val, "test": our_test}

    if cfg.get("strict_tile_cap", False):
        before = {k: len(v) for k, v in splits.items()}
        splits = apply_strict_tile_cap(
            splits,
            {"train": cfg["n_train"], "val": cfg["n_val"], "test": cfg["n_test"]},
        )
        print(f"strict_tile_cap applied: before={before} after={[ (k,len(v)) for k,v in splits.items()]}")

    # Enforce per field caps after strict cut as well (last field may be partial)
    for split_name, cap in caps.items():
        if cap is None:
            continue
        by_f: dict[str, list[str]] = defaultdict(list)
        for t in splits[split_name]:
            by_f[field_id_from_name(t)].append(t)
        rebuilt: list[str] = []
        for fid, flist in by_f.items():
            if len(flist) > int(cap):
                raise RuntimeError(
                    f"split {split_name} field {fid} has {len(flist)} tiles > cap {cap}"
                )
            rebuilt.extend(flist)
        splits[split_name] = rebuilt

    assert_no_field_overlap(splits)

    rows: list[dict[str, str]] = []
    for our_split, tiles in splits.items():
        source = "train" if our_split in ("train", "val") else "val"
        for t in tiles:
            rows.append(
                {
                    "tile_id": t,
                    "source_split": source,
                    "our_split": our_split,
                    "field_id": field_id_from_name(t),
                }
            )
    write_split_csv(split_csv, rows)
    print(f"Wrote {split_csv} rows={len(rows)}")
    fields_per = {
        sp: len({r["field_id"] for r in rows if r["our_split"] == sp})
        for sp in ("train", "val", "test")
    }
    print(
        f"counts train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} "
        f"fields_train={fields_per['train']} fields_val={fields_per['val']} "
        f"fields_test={fields_per['test']}"
    )

    # Per field size check
    for sp, tiles in splits.items():
        sizes = defaultdict(int)
        for t in tiles:
            sizes[field_id_from_name(t)] += 1
        cap = caps.get(sp)
        if cap is not None and sizes and max(sizes.values()) > int(cap):
            raise RuntimeError(f"{sp} exceeds max_tiles_per_field={cap}")

    print(
        "Plan note: per class tile counts and overlap pixels need labels on disk. "
        "Run extract then compute_stats.py for those numbers. "
        "Warn threshold: any class with fewer than 10 tiles in a split."
    )

    if args.mode == "plan":
        return

    if args.mode == "smoke5":
        rows = rows[:5]
        print("SMOKE: extracting 5 tiles only")

    extract_subset(archive, subset_root, rows)
    print("Done.")


if __name__ == "__main__":
    main()
