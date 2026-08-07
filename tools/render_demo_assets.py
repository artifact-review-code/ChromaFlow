#!/usr/bin/env python3
"""Render deterministic demo crops from source images and instance maps."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
ROOT=Path(__file__).resolve().parents[1]; SOURCE=ROOT/"tmp"/"demo"; OUTPUT=ROOT/"assets"/"demo"
SAMPLES={"MO":[182,390,420],"CPM":[26,0,420],"CONIC":[0,0,256],"NUINSSEG":[78,0,420]}
def colorize(labels):
    out=np.zeros((*labels.shape,3),dtype=np.uint8)
    for i in np.unique(labels[labels>0]):out[labels==i]=np.random.default_rng(int(i)*104729).integers(45,245,3,dtype=np.uint8)
    return out
def main():
    for dataset,(x,y,size) in SAMPLES.items():
        image=np.asarray(Image.open(SOURCE/f"{dataset}_input.png").convert("RGB"))[y:y+size,x:x+size]
        labels=np.load(SOURCE/f"{dataset}_prediction.npy")[y:y+size,x:x+size]
        foreground=labels>0
        boundary=foreground ^ ndimage.binary_erosion(foreground,structure=np.ones((3,3),dtype=bool))
        overlay=image.copy(); overlay[boundary]=[220,30,45]; target=OUTPUT/dataset; target.mkdir(parents=True,exist_ok=True)
        Image.fromarray(image).save(target/"input.png",optimize=True); Image.fromarray(colorize(labels)).save(target/"prediction.png",optimize=True); Image.fromarray(overlay).save(target/"overlay.png",optimize=True); np.save(target/"instances.npy",labels.astype(np.int32))
    manifest=json.loads((OUTPUT/"manifest.json").read_text())
    for item in manifest["samples"]:
        labels=np.load(OUTPUT/item["dataset"]/"instances.npy")
        item["predicted_instances"]=int(np.count_nonzero(np.unique(labels)))
    (OUTPUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
if __name__=="__main__":main()
