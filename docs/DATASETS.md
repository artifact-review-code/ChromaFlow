# Datasets and final split manifests

| Dataset | Original images/patches | Train / validation / test | Magnification |
|---|---:|---:|---|
| MoNuSeg | 44 | 24 / 6 / 14 | 40x |
| CPM-17 | 64 (32 official development + 32 official test) | 24 / 8 / 32 | 20x or 40x |
| CoNIC | 4981 | 3372 / 516 / 1093 | 20x |
| NuInsSeg | 665 | 472 / 65 / 128 | 40x |

The counts above are the **current manuscript protocol**. The old CPM-17,
CoNIC and NuInsSeg manifests have been removed at the author's request.
Supply the final experiment manifests externally; the program never invents
membership or randomly samples an evaluation subset. MoNuSeg's retained
manifest is in [splits/monuseg.json](../splits/monuseg.json); its TCGA barcode
prefixes identify participants.

Place each manifest at `<DATA_ROOT>/splits/{monuseg,cpm17,conic,nuinsseg}.json`.
Alternatively use `data/official_clean/core/<dataset>/splits/final.json`.
Training preparation and full-image evaluation use the **same** file.
The schema is:

```json
{
  "train": ["image_a"],
  "val": ["image_b"],
  "test": ["image_c"],
  "groups": {
    "image_a": {"patient_id": "patient_a", "wsi_id": "slide_a"},
    "image_b": {"patient_id": "patient_b", "wsi_id": "slide_b"},
    "image_c": {"patient_id": "patient_c", "wsi_id": "slide_c"}
  }
}
```

These three IDs illustrate the schema only. Real files must contain the counts
in the table. Obtain patient and slide IDs from source metadata. All patches
from a patient, including adjacent patches and multiple slides, belong to one
split. Validation checks counts, duplicate images, patient overlap and WSI
overlap. For MoNuSeg, patient IDs can be derived from the TCGA barcode.

Splits are applied **before** point sampling and patch extraction. Prepared
patches retain their original image ID, coordinates and split-manifest SHA256.
A changed manifest requires new preprocessing. Source validation selects the
checkpoint; that checkpoint and decoder are reused across target datasets.

Sources: [MoNuSeg](https://monuseg.grand-challenge.org/Data/),
[CPM-17](https://github.com/nauyan/NucleiSegmentation),
[CoNIC](https://conic-challenge.grand-challenge.org/),
[NuInsSeg](https://www.kaggle.com/datasets/ipateam/nuinsseg).
Dataset licenses remain with the original providers.
