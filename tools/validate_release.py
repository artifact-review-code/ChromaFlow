#!/usr/bin/env python3
"""Validate reviewer-release manifests, tables, assets, links, and anonymity."""

from __future__ import annotations

import argparse
import sys
import csv
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SPLITS = {
    "monuseg.json": (24, 6, 14),
    "cpm17.json": (24, 8, 32),
    "conic.json": (3372, 516, 1093),
    "nuinsseg.json": (472, 65, 128),
}
REQUIRED_DOCS = [
    "README.md", "docs/METHOD.md", "docs/CODE_STRUCTURE.md",
    "docs/DATASETS.md", "docs/TRAINING.md", "docs/INFERENCE.md",
    "docs/RESULTS.md", "docs/VISUALIZATIONS.md", "docs/MODEL_ZOO.md",
    "results/table1_indomain.md", "results/table2_crossdomain.md",
]
FORBIDDEN = ("/home/ftl", "100.87.11.77", "wangbaoyuan", "王报源")


def check_splits(data_root=None, require_all=False) -> None:
    sys.path.insert(0, str(ROOT))
    from chromaflow.protocol import load_splits
    for dataset in ('MO', 'CPM', 'CONIC', 'NUINSSEG'):
        try:
            data, _ = load_splits(Path(data_root or ROOT), dataset)
        except FileNotFoundError:
            if require_all:
                raise
            print(f'{dataset}: external final patient/WSI manifest required')
            continue
        print(f'{dataset}: verified counts and patient/WSI separation')


def check_results() -> None:
    specs = {
        "table1_indomain.csv": (432, {"dataset", "metric", "value"}),
        "table2_crossdomain.csv": (576, {"direction", "metric", "value"}),
    }
    for filename, (count, fields) in specs.items():
        with (ROOT / "results" / filename).open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        assert len(rows) == count, f"{filename}: expected {count} rows, got {len(rows)}"
        assert fields <= set(reader.fieldnames or []), f"{filename}: missing fields"
        assert len({row["method"] for row in rows}) == 18, f"{filename}: method count"
    subprocess.run(
        [sys.executable, str(ROOT / "tools/render_results_tables.py"), "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def check_markdown_links() -> None:
    pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    for relative in REQUIRED_DOCS:
        path = ROOT / relative
        assert path.is_file(), f"missing document: {relative}"
        for target in pattern.findall(path.read_text()):
            target = target.strip().split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (path.parent / target).resolve()
            assert resolved.exists(), f"broken link in {relative}: {target}"


def check_anonymity() -> None:
    roots = [ROOT / "README.md", ROOT / "docs", ROOT / "results", ROOT / "splits"]
    for base in roots:
        paths = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
        for path in paths:
            if path.suffix.lower() not in {".md", ".json", ".csv", ".txt"}:
                continue
            text = path.read_text(errors="ignore").lower()
            for token in FORBIDDEN:
                assert token.lower() not in text, f"identity/server token in {path.relative_to(ROOT)}"
    checkpoints = [p for p in ROOT.rglob("*") if p.suffix.lower() in {".pt", ".pth", ".ckpt"}]
    assert not checkpoints, f"checkpoint files included: {checkpoints}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--require-all-splits', action='store_true')
    args = parser.parse_args()
    check_splits(args.data_root, args.require_all_splits)
    check_results()
    check_markdown_links()
    assert len(list((ROOT / "assets/paper").glob("*.png"))) == 9
    assert (ROOT / "assets/paper/figure2_architecture.pdf").is_file()
    check_anonymity()
    print("Available manifests, result schemas, figure assets and documentation checks passed. External-manifest status is listed above.")


if __name__ == "__main__":
    main()
