#!/usr/bin/env python3
"""Validate reviewer-release manifests, tables, assets, links, and anonymity."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SPLITS = {
    "monuseg.json": (24, 6, 14),
    "cpm17.json": (20, 4, 8),
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


def check_splits() -> None:
    for filename, expected in EXPECTED_SPLITS.items():
        data = json.loads((ROOT / "splits" / filename).read_text())
        groups = [data[key] for key in ("train", "val", "test")]
        actual = tuple(len(group) for group in groups)
        assert actual == expected, f"{filename}: expected {expected}, got {actual}"
        normalized = [{str(item) for item in group} for group in groups]
        assert not (normalized[0] & normalized[1]), f"{filename}: train/val overlap"
        assert not (normalized[0] & normalized[2]), f"{filename}: train/test overlap"
        assert not (normalized[1] & normalized[2]), f"{filename}: val/test overlap"


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
        ["python3", str(ROOT / "tools/render_results_tables.py"), "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def check_demo() -> None:
    manifest = json.loads((ROOT / "assets/demo/manifest.json").read_text())
    assert len(manifest["samples"]) == 4, "demo manifest must contain four samples"
    for sample in manifest["samples"]:
        folder = ROOT / "assets/demo" / sample["dataset"]
        for filename in ("input.png", "prediction.png", "overlay.png", "instances.npy"):
            assert (folder / filename).is_file(), f"missing demo asset: {folder / filename}"


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
    roots = [ROOT / "README.md", ROOT / "docs", ROOT / "results", ROOT / "splits", ROOT / "assets/demo"]
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
    check_splits()
    check_results()
    check_demo()
    check_markdown_links()
    check_anonymity()
    print("Release validation passed: splits, results, demos, links, and anonymity.")


if __name__ == "__main__":
    main()
