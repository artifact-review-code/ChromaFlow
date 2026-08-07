# Model zoo

No model weights are distributed during anonymous review. The rows below reserve stable names and intended configurations so that release URLs, checksums, and file sizes can be added without changing the inference documentation.

| Planned checkpoint | Architecture | Training domain | Recommended use | Download | SHA-256 / size |
|---|---|---|---|---|---|
| `chromaflow_mo_r34` | ResNet-34 U-Net | MoNuSeg | MoNuSeg-like 40x multi-organ H&E | To be released upon acceptance | Pending |
| `chromaflow_cpm_r34` | ResNet-34 U-Net | CPM-17 | CPM-17-like cancer H&E | To be released upon acceptance | Pending |
| `chromaflow_conic_r34` | ResNet-34 U-Net | CoNIC | 20x colorectal H&E patches | To be released upon acceptance | Pending |
| `chromaflow_nuinsseg_r34` | ResNet-34 U-Net | NuInsSeg | multi-organ NuInsSeg-style patches | To be released upon acceptance | Pending |

All planned models use a scalar-potential flow head and the canonical 224/64 tiled inference configuration. A checkpoint should only be considered official after this page contains a working URL and a SHA-256 checksum. See the [Inference guide](INFERENCE.md) for loading and decoding.
