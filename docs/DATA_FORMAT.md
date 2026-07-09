# Data Format

The repository does not include datasets. `--root` should point to a directory containing `data/official_clean`.

## Training Patches

```text
data/official_clean/
  bonus_prepared/{MO,CPM}/images/{train,val,test}/{tile}.png
  bonus_prepared/{MO,CPM}/ratio2.0/labels_clu/{train,val,test}/{tile}_label_clu.png
  ours_prepared/{MO,CPM}/ratio2.0/pob_targets/{train,val,test}/{tile}.npz
```

Each `pob_targets` NPZ file must contain:

- `offset`: float32 array with shape `(2, H, W)`.
- `basin_valid`: binary/float array with shape `(H, W)`.
- `point_map`: binary point annotation map with shape `(H, W)`.
- `instance_id_soft`: int array with shape `(H, W)`.

The default experiments use 224x224 patches. The loader shuffles all prepared patches each epoch.

## Full-Image Evaluation

```text
data/official_clean/core/
  monuseg/
    images/{image_id}.png
    labels_instance/{image_id}.npy
    splits/bonus_official_ids.json
  cpm17/
    images/{image_id}.png
    labels_instance/{image_id}.npy
    splits/bonus_official_arranged_ids.json
    mappings/bonus_arranged_to_official_train.json
```

Instance labels are integer maps with `0` as background and positive values as nuclei IDs.
