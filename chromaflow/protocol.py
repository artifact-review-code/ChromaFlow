"""Manuscript protocol and verified, patient-disjoint data membership."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

DATASETS = {'MO': 'monuseg', 'CPM': 'cpm17', 'CONIC': 'conic', 'NUINSSEG': 'nuinsseg'}
EXPECTED_COUNTS = {'MO': (24, 6, 14), 'CPM': (24, 8, 32),
                   'CONIC': (3372, 516, 1093), 'NUINSSEG': (472, 65, 128)}
LOSS_WEIGHTS = dict(fg=1.0, flow=2.0, point=2.0, div=0.5, center=0.75, H=1.0, E=0.5)
PROTOCOL_VERSION = 'chromaflow-tmi-2026-10-21'


def validate_splits(data: dict, dataset: str, check_counts: bool = True) -> None:
    groups = [data[k] for k in ('train', 'val', 'test')]
    if any(len(g) != len(set(g)) for g in groups):
        raise ValueError('Duplicate image IDs within a split')
    if check_counts and tuple(map(len, groups)) != EXPECTED_COUNTS[dataset]:
        raise ValueError(f'{dataset}: expected train/val/test {EXPECTED_COUNTS[dataset]}, got {tuple(map(len, groups))}')
    memberships = [{str(x) for x in g} for g in groups]
    for i in range(3):
        for j in range(i):
            if memberships[i] & memberships[j]:
                raise ValueError('Image overlap between splits')
    records = data.get('groups', {})
    patient_seen, wsi_seen = {}, {}
    for split, group in zip(('train', 'val', 'test'), groups):
        for name in group:
            record = records.get(name, {})
            patient = record.get('patient_id')
            wsi = record.get('wsi_id')
            # TCGA barcodes encode the participant in the first three components.
            if dataset == 'MO' and re.match(r'^TCGA-[^-]+-[^-]+-', name):
                patient = patient or '-'.join(name.split('-')[:3])
                wsi = wsi or name
            if not patient or not wsi:
                raise ValueError(f'{name}: supply groups[image_id].patient_id and .wsi_id from source metadata')
            for key, seen in ((patient, patient_seen), (wsi, wsi_seen)):
                if key in seen and seen[key] != split:
                    raise ValueError(f'Patient/WSI {key} appears in {seen[key]} and {split}')
                seen[key] = split


def split_path(root: Path, dataset: str) -> Path:
    """External final manifests override the verified bundled MoNuSeg manifest."""
    filename = DATASETS[dataset] + '.json'
    for path in (root / 'splits' / filename,
                 root / 'data/official_clean/core' / DATASETS[dataset] / 'splits/final.json'):
        if path.is_file():
            return path
    if dataset == 'MO':
        return Path(__file__).resolve().parents[1] / 'splits/monuseg.json'
    raise FileNotFoundError(f'Final {dataset} split required: {root / "splits" / filename}. See docs/DATASETS.md.')


def load_splits(root: Path, dataset: str) -> tuple[dict, str]:
    path = split_path(Path(root), dataset)
    raw = path.read_bytes()
    data = json.loads(raw)
    validate_splits(data, dataset)
    return data, hashlib.sha256(raw).hexdigest()


def dataset_base(root: Path, dataset: str) -> Path:
    return Path(root) / 'data/official_clean/core' / DATASETS[dataset]
