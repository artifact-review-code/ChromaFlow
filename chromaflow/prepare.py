"""Prepare fixed point-only targets before patch extraction.

The only use of instance masks is optional point extraction (one EDT peak per
instance). All subsequent supervision uses RGB and the retained point positions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from skimage.morphology import disk
from chromaflow.protocol import PROTOCOL_VERSION, load_splits, dataset_base


def instance_points(mask: np.ndarray) -> np.ndarray:
    points = np.zeros(mask.shape, np.uint8)
    for uid in np.unique(mask):
        if uid == 0:
            continue
        dist = distance_transform_edt(mask == uid)
        points[np.unravel_index(np.argmax(dist), dist.shape)] = 1
    return points


def retain_points(point_map: np.ndarray, ratio: float, seed: int, image_id: str) -> np.ndarray:
    if not 0 < ratio <= 1:
        raise ValueError('point ratio must be in (0, 1]')
    coords = np.argwhere(point_map > 0)
    # Round to nearest integer; retain one point in a nonempty small image.
    n = min(len(coords), max(1, int(np.floor(len(coords) * ratio + 0.5)))) if len(coords) else 0
    digest = hashlib.sha256(f'{seed}:{image_id}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    out = np.zeros(point_map.shape, np.uint8)
    if n:
        chosen = coords[rng.choice(len(coords), n, replace=False)]
        out[chosen[:, 0], chosen[:, 1]] = 1
    return out


def shift_points(point_map: np.ndarray, radius: int, seed: int, image_id: str) -> np.ndarray:
    """Fixed uniform integer displacement within a Euclidean radius, before tiling.

    Reject out-of-image and occupied positions to preserve point count.
    """
    if radius < 0:
        raise ValueError('point shift must be nonnegative')
    if radius == 0:
        return point_map.copy()
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'shift:{seed}:{image_id}'.encode()).digest()[:8],'little'))
    offsets=np.asarray([(dy,dx) for dy in range(-radius,radius+1) for dx in range(-radius,radius+1) if dy*dy+dx*dx<=radius*radius])
    out=np.zeros_like(point_map);h,w=out.shape
    for y,x in np.argwhere(point_map>0):
        candidates=offsets+np.array([y,x]);valid=(candidates[:,0]>=0)&(candidates[:,0]<h)&(candidates[:,1]>=0)&(candidates[:,1]<w)
        candidates=candidates[valid];candidates=candidates[out[candidates[:,0],candidates[:,1]]==0]
        if not len(candidates):raise ValueError('No collision-free shifted point location')
        ny,nx=candidates[rng.integers(len(candidates))];out[ny,nx]=1
    return out


def point_targets(point_map: np.ndarray, scale: float = 250.0) -> dict:
    h, w = point_map.shape
    coords = np.argwhere(point_map > 0)
    if not len(coords):
        return dict(offset=np.zeros((2,h,w),np.float32), basin_valid=np.zeros((h,w),np.float32),
                    point_map=point_map.astype(np.uint8), instance_id_soft=np.zeros((h,w),np.int32),
                    rho=np.zeros((h,w),np.float32), center_target=np.zeros((h,w),np.float32))
    yy, xx = np.indices((h,w)); query=np.stack((yy,xx),axis=-1)
    distance, nearest = cKDTree(coords).query(query)
    delta = coords[nearest] - query
    offset = np.stack((delta[...,1],delta[...,0])).astype(np.float32)/scale
    density = np.exp(-distance**2/(2*3.0**2)).astype(np.float32)
    return dict(offset=offset, basin_valid=np.ones((h,w),np.float32), point_map=point_map.astype(np.uint8),
                instance_id_soft=(nearest+1).astype(np.int32), rho=density, center_target=density.copy())


def cluster_labels(image: np.ndarray, point_map: np.ndarray, seed: int = 0) -> np.ndarray:
    """Three-class RGB/distance K-means inside each point Voronoi cell.

    Implementation convention: foreground is the cluster most represented in
    the radius-2 point disk; background is the remaining cluster with the largest
    mean point distance; the third cluster is ignored. Voronoi boundaries are
    dilated by disk(2) into an ignore band, and radius-2 point disks are foreground.
    Empty-point images are ignored rather than treated as dense negative labels.
    """
    targets = point_targets(point_map)
    cells = targets['instance_id_soft']; out = np.full(point_map.shape,2,np.uint8)
    points = np.argwhere(point_map > 0)
    if not len(points):
        return out
    yy,xx=np.indices(point_map.shape)
    for i,(y,x) in enumerate(points,1):
        cell=cells==i; distance=np.hypot(yy[cell]-y,xx[cell]-x)
        features=np.column_stack((image[cell].astype(np.float64)/255., np.minimum(distance/250.,1.)))
        if len(features)<3 or len(np.unique(features,axis=0))<3:
            out[cell]=2;continue
        pred=KMeans(n_clusters=3,random_state=seed,n_init=10).fit_predict(features)
        near=distance<=2
        fg=int(np.bincount(pred[near],minlength=3).argmax())
        rest=[j for j in range(3) if j!=fg]
        bg=max(rest,key=lambda j: float(distance[pred==j].mean()) if np.any(pred==j) else -1)
        labels=np.full(len(pred),2,np.uint8);labels[pred==fg]=1;labels[pred==bg]=0;out[cell]=labels
    boundary=np.zeros_like(cells,bool)
    boundary[1:] |= cells[1:] != cells[:-1];boundary[:-1] |= cells[1:] != cells[:-1]
    boundary[:,1:] |= cells[:,1:] != cells[:,:-1];boundary[:,:-1] |= cells[:,1:] != cells[:,:-1]
    out[binary_dilation(boundary,structure=disk(2))]=2
    out[binary_dilation(point_map>0,structure=disk(2))]=1
    out[np.all(image==0,axis=-1) & (point_map==0)]=0
    return out


def starts(length: int, size: int, stride: int) -> list[int]:
    return list(dict.fromkeys(list(range(0,max(1,length-size+1),stride))+[max(0,length-size)]))


def prepare(root: Path, dataset: str, ratio: float=1., seed: int=0, stride: int=250,
            generate_points: bool=False, point_shift: int=0) -> Path:
    if stride <= 0 or stride > 250: raise ValueError('stride must be in [1, 250]')
    splits, split_hash=load_splits(root,dataset);base=dataset_base(root,dataset)
    bonus=root/'data/official_clean/bonus_prepared'/dataset
    ours=root/'data/official_clean/ours_prepared'/dataset
    manifest_path=ours/'preparation.json'
    if manifest_path.exists() or (bonus/'images/train').exists() or (ours/'ratio2.0/pob_targets/train').exists():
        raise FileExistsError('Prepared outputs already exist. Use a separate data root for each point budget/seed.')
    records={};selection={};patch_size=250
    for name in splits['train']:
        image=np.asarray(Image.open(base/'images'/f'{name}.png').convert('RGB'))
        if generate_points:
            pm=instance_points(np.load(base/'labels_instance'/f'{name}.npy'))
        else:
            pm=np.load(base/'points'/f'{name}.npy')
        if pm.shape!=image.shape[:2]:raise ValueError(f'{name}: image/point shape mismatch')
        kept=shift_points(retain_points(pm,ratio,seed,name),point_shift,seed,name)
        selection[name]={'available':int((pm>0).sum()),'retained_yx':np.argwhere(kept>0).tolist()}
        # Construct every target from retained original-image points, then tile.
        target=point_targets(kept);clu=cluster_labels(image,kept,seed)
        h,w=kept.shape
        for y in starts(h,patch_size,stride):
            for x in starts(w,patch_size,stride):
                tile=f'{name}__y{y}_x{x}';sh=min(patch_size,h-y);sw=min(patch_size,w-x)
                rgb=np.full((patch_size,patch_size,3),255,np.uint8);rgb[:sh,:sw]=image[y:y+sh,x:x+sw]
                labels=np.full((patch_size,patch_size),2,np.uint8);labels[:sh,:sw]=clu[y:y+sh,x:x+sw]
                npz={}
                for key,value in target.items():
                    out=np.zeros((*value.shape[:-2],patch_size,patch_size),value.dtype)
                    out[...,:sh,:sw]=value[...,y:y+sh,x:x+sw];npz[key]=out
                # Valid only where weak foreground is present, with no instance GT.
                npz['basin_valid']=(labels==1).astype(np.float32)
                encoded=np.asarray([[255,0,0],[0,255,0],[0,0,255]],np.uint8)[labels]
                ip=bonus/'images/train'/f'{tile}.png';cp=bonus/'ratio2.0/labels_clu/train'/f'{tile}_label_clu.png';tp=ours/'ratio2.0/pob_targets/train'/f'{tile}.npz'
                for path in (ip,cp,tp):path.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(rgb).save(ip);Image.fromarray(encoded).save(cp);np.savez_compressed(tp,**npz)
                records[tile]={'image_id':name,'origin_yx':[y,x]}
    manifest={'protocol':PROTOCOL_VERSION,'dataset':dataset,'split_sha256':split_hash,
              'point_ratio':ratio,'point_subset_seed':seed,'point_shift':point_shift,'patch_size':250,'flow_scale':250.,
              'sampling':'uniform_without_replacement_once_per_original_image',
              'patch_stride':stride,'point_selection':selection,'tiles':records}
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    manifest_path.write_text(json.dumps(manifest,indent=2));return manifest_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--dataset',choices=['MO','CPM','CONIC','NUINSSEG'],required=True)
    p.add_argument('--point-ratio',type=float,default=1.);p.add_argument('--point-subset-seed',type=int,default=0)
    p.add_argument('--stride',type=int,default=250);p.add_argument('--generate-points-from-masks',action='store_true')
    p.add_argument('--point-shift',type=int,default=0)
    a=p.parse_args();print(prepare(a.root,a.dataset,a.point_ratio,a.point_subset_seed,a.stride,a.generate_points_from_masks,a.point_shift))
if __name__=='__main__':main()
