# Implementation Details

Default experiments use a ResNet-34 U-Net style encoder-decoder with foreground, centerness, and flow heads. RGB inputs are converted from uint8 to `[0, 1]` and normalized with ImageNet mean/std. Hematoxylin/eosin channels are computed with `skimage.color.rgb2hed` and robustly normalized per image using 1/99 percentiles.

Models are trained for 150 epochs with Adam, learning rate `1e-4`, weight decay `3e-4`, betas `(0.9, 0.99)`, default epsilon `1e-8`, and a constant learning rate. No LR warmup or early stopping is used in the default scripts.

The final checkpoint is selected by best AJI on the validation split, evaluated every 10 epochs. The released scripts explicitly pass `--ckpt-split val`.

Training uses prepared 224x224 patches. Each epoch iterates over all prepared training patches with dataloader shuffling. Geometric augmentation includes random horizontal/vertical flips and random 90-degree rotations; optional HED stain augmentation is enabled in the default scripts.

Runtime and FLOP measurements should state the input size, batch size, hardware, whether preprocessing and Cellpose-style decoding are included, and whether CUDA synchronization and warmup iterations were used.
