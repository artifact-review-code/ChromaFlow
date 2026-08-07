#!/usr/bin/env python3
"""Render the tidy result CSV files as GitHub-readable wide Markdown tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GROUP_NAMES = {
    "fully_supervised": "Fully supervised references",
    "foundation_model": "Pretrained foundation-model references",
    "weakly_supervised": "Weakly supervised methods",
}
MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}


def weak_ranks(rows: list[dict[str, str]], key_fields: tuple[str, ...]) -> dict[tuple[str, ...], int]:
    """Rank non-missing weak-supervision values independently per result column."""
    columns: dict[tuple[str, ...], list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        if row["group"] == "weakly_supervised" and row["value"]:
            column = tuple(row[field] for field in key_fields)
            columns[column].append((row["method"], float(row["value"])))
    result = {}
    for column, values in columns.items():
        distinct = sorted({value for _, value in values}, reverse=True)[:3]
        for method, value in values:
            if value in distinct:
                result[(method, *column)] = distinct.index(value) + 1
    return result


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] + ["---:" for _ in headers[1:]]) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_indomain() -> str:
    source = ROOT / "results/table1_indomain.csv"
    rows = list(csv.DictReader(source.open(newline="")))
    datasets = ["MoNuSeg", "CPM-17", "CoNIC", "NuInsSeg"]
    metrics = ["AJI", "Dice", "DQ", "SQ", "PQ", "Recall"]
    lookup = {(r["method"], r["dataset"], r["metric"]): r["value"] or "—" for r in rows}
    ranks = weak_ranks(rows, ("dataset", "metric"))
    meta = {}
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        meta[row["method"]] = (row["params_m"], row["gflops"])
        if row["method"] not in groups[row["group"]]:
            groups[row["group"]].append(row["method"])
    headers = ["Method", "Params (M)", "GFLOPs"] + [f"{d} {m}" for d in datasets for m in metrics]
    parts = ["# Table 1 — in-domain results", "", "All values are copied from `table1_indomain.csv`. Within weak supervision, 🥇/🥈/🥉 mark the best/second/third value in each column. Missing values were not reported by the cited source."]
    for group, methods in groups.items():
        body = []
        for method in methods:
            values = []
            for dataset in datasets:
                for metric in metrics:
                    value = lookup[(method, dataset, metric)]
                    medal = MEDALS.get(ranks.get((method, dataset, metric)), "")
                    values.append(f"{medal}{value}")
            body.append([method, *meta[method], *values])
        parts.extend(["", f"## {GROUP_NAMES[group]}", "", markdown_table(headers, body)])
    return "\n".join(parts) + "\n"


def render_crossdomain() -> str:
    source = ROOT / "results/table2_crossdomain.csv"
    rows = list(csv.DictReader(source.open(newline="")))
    directions = ["MO->CPM", "CPM->MO", "Co->Nu", "Nu->Co", "MO->Co", "MO->Nu", "CPM->Co", "CPM->Nu"]
    metrics = ["AJI", "Dice", "DQ", "PQ"]
    lookup = {(r["method"], r["direction"], r["metric"]): r["value"] or "—" for r in rows}
    ranks = weak_ranks(rows, ("direction", "metric"))
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["method"] not in groups[row["group"]]:
            groups[row["group"]].append(row["method"])
    headers = ["Method"] + [f"{d} {m}" for d in directions for m in metrics]
    parts = ["# Table 2 — cross-domain results", "", "All values are copied from `table2_crossdomain.csv`. Within weak supervision, 🥇/🥈/🥉 mark the best/second/third value in each column. Foundation-model target results repeat because those models are not retrained for each source."]
    for group, methods in groups.items():
        body = []
        for method in methods:
            values = []
            for direction in directions:
                for metric in metrics:
                    value = lookup[(method, direction, metric)]
                    medal = MEDALS.get(ranks.get((method, direction, metric)), "")
                    values.append(f"{medal}{value}")
            body.append([method, *values])
        parts.extend(["", f"## {GROUP_NAMES[group]}", "", markdown_table(headers, body)])
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if rendered tables differ")
    args = parser.parse_args()
    targets = {
        ROOT / "results/table1_indomain.md": render_indomain(),
        ROOT / "results/table2_crossdomain.md": render_crossdomain(),
    }
    for path, content in targets.items():
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"Rendered result table drift detected: {path.relative_to(ROOT)}")
            continue
        path.write_text(content)
        print(f"wrote {path.relative_to(ROOT)}")
    if args.check:
        print("Rendered Markdown tables match the result CSV files.")


if __name__ == "__main__":
    main()
