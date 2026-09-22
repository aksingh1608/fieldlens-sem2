#!/usr/bin/env python3
"""Walk Agriculture-Vision S3 prefix one level at a time; print tree, counts, sizes, samples."""

from __future__ import annotations

import argparse
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BUCKET_HOST = "https://intelinair-data-releases.s3.amazonaws.com/"
DEFAULT_PREFIX = "agriculture-vision/cvpr_challenge_2021/supervised/"


def _local(tag: str) -> str:
    """Strip XML namespace from an Element tag."""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _children(parent: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in list(parent) if _local(c.tag) == name]


def _child_text(parent: ET.Element, name: str, default: str = "") -> str:
    for c in list(parent):
        if _local(c.tag) == name:
            return c.text or default
    return default


def list_level(prefix: str, max_keys: int = 1000) -> tuple[list[str], list[tuple[str, int]]]:
    """Return (common_prefixes, [(key, size), ...]) for one delimiter level."""
    dirs: list[str] = []
    files: list[tuple[str, int]] = []
    continuation: str | None = None
    while True:
        params: dict[str, str] = {
            "list-type": "2",
            "delimiter": "/",
            "prefix": prefix,
            "max-keys": str(max_keys),
        }
        if continuation:
            params["continuation-token"] = continuation
        url = BUCKET_HOST + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=60) as resp:
            root = ET.fromstring(resp.read())
        for cp in _children(root, "CommonPrefixes"):
            p = _child_text(cp, "Prefix")
            if p:
                dirs.append(p)
        for content in _children(root, "Contents"):
            key = _child_text(content, "Key")
            size_s = _child_text(content, "Size", "0")
            if key and key != prefix:
                files.append((key, int(size_s)))
        is_truncated = _child_text(root, "IsTruncated", "false")
        if is_truncated.lower() != "true":
            break
        continuation = _child_text(root, "NextContinuationToken") or None
        if not continuation:
            break
    return dirs, files


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{n} B"


ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".zip", ".7z")


def walk(prefix: str, max_depth: int, depth: int = 0) -> None:
    indent = "  " * depth
    dirs, files = list_level(prefix)
    total_size = sum(s for _, s in files)
    archive_hits = [k for k, _ in files if k.lower().endswith(ARCHIVE_SUFFIXES)]
    print(f"{indent}{prefix}")
    print(f"{indent}  dirs={len(dirs)} files={len(files)} size={human_bytes(total_size)}")
    if archive_hits:
        print(f"{indent}  ARCHIVES_DETECTED={len(archive_hits)}")
        for k in archive_hits[:10]:
            print(f"{indent}    archive: {k}")
    samples = [k.rsplit("/", 1)[-1] or k for k, _ in files[:5]]
    if samples:
        print(f"{indent}  samples: {samples}")
    for k, s in files[:20]:
        name = k[len(prefix) :] if k.startswith(prefix) else k
        print(f"{indent}  - {name} ({human_bytes(s)})")
    if depth >= max_depth:
        for d in dirs:
            print(f"{indent}  [skip deeper] {d}")
        return
    for d in dirs:
        walk(d, max_depth, depth + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="S3 key prefix under intelinair-data-releases",
    )
    parser.add_argument("--max-depth", type=int, default=3)
    args = parser.parse_args()
    print("bucket=intelinair-data-releases")
    print(f"prefix={args.prefix}")
    print("---")
    walk(args.prefix, args.max_depth)


if __name__ == "__main__":
    main()
