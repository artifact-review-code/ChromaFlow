import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from chromaflow.protocol import validate_splits, load_splits
from chromaflow.prepare import retain_points, point_targets, cluster_labels, instance_points, prepare
from chromaflow.dataset import NucleiV3PatchDataset, _scale_jitter, stain_h_channel
from chromaflow.model import potential_to_flow, NISFlowNet
from chromaflow.losses import (divergence_sink_loss, masked_mse, compute_losses,
    compute_stain_anchored_soft_label_loss, compute_eosin_anchored_neg_loss)

def test_fixed_sampling_exact_budget_and_no_replacement():
    pm=np.zeros((250,250),np.uint8);pm[5:245:10,5:245:10]=1
    keep=retain_points(pm,.05,8,'image-a')
    assert keep.sum()==int(np.floor(pm.sum()*.05+.5))
    assert np.array_equal(keep,retain_points(pm,.05,8,'image-a'))
    assert not np.any(keep & ~pm)
    assert not np.array_equal(keep,retain_points(pm,.05,9,'image-a'))
    assert retain_points(np.zeros((8,8)),.05,8,'empty').sum()==0


def test_retained_only_targets_and_fixed_displacement_scale():
    pm=np.zeros((250,250),np.uint8);pm[20,25]=1;pm[100,100]=1
    kept=np.zeros_like(pm);kept[20,25]=1;t=point_targets(kept)
    np.testing.assert_allclose(t['offset'][:,20,50],[-.1,0])
    assert t['rho'][20,25]==1 and t['rho'][100,100]<1e-10
    assert np.array_equal(t['center_target'],t['rho'])
    assert t['instance_id_soft'].max()==1
    # Cropping preserves the contribution of a point lying just outside the crop.
    assert t['rho'][21,25]>0


def test_cluster_maps_are_point_only_and_empty_points_are_ignored():
    rgb=np.full((32,32,3),220,np.uint8);rgb[8:20,8:20]=[100,30,110]
    pm=np.zeros((32,32),np.uint8);pm[14,14]=1
    labels=cluster_labels(rgb,pm)
    assert labels[14,14]==1
    assert set(np.unique(labels)) <= {0,1,2}
    assert np.all(cluster_labels(rgb,np.zeros_like(pm))==2)


def test_patient_and_slide_leakage_rejected():
    data={'train':['a'],'val':['b'],'test':['c'], 'groups':{
        'a':{'patient_id':'p1','wsi_id':'w1'},'b':{'patient_id':'p2','wsi_id':'w2'},
        'c':{'patient_id':'p3','wsi_id':'w3'}}}
    validate_splits(data,'CPM',False)
    bad=copy.deepcopy(data);bad['groups']['b']['patient_id']='p1'
    with pytest.raises(ValueError,match='Patient/WSI'):validate_splits(bad,'CPM',False)
    bad=copy.deepcopy(data);bad['groups']['b']['wsi_id']='w1'
    with pytest.raises(ValueError,match='Patient/WSI'):validate_splits(bad,'CPM',False)
    del data['groups']['c']
    with pytest.raises(ValueError,match='patient_id'):validate_splits(data,'CPM',False)


def test_bundled_monuseg_patient_disjoint_and_missing_others_fail(tmp_path):
    data,_=load_splits(tmp_path,'MO');assert [len(data[k]) for k in ('train','val','test')]==[24,6,14]
    for ds in ('CPM','CONIC','NUINSSEG'):
        with pytest.raises(FileNotFoundError):load_splits(tmp_path,ds)


def test_source_residual_and_flow_error_normalization():
    y,x=torch.meshgrid(torch.arange(9.),torch.arange(9.),indexing='ij')
    phi=(x*x+y*y)[None,None];v=potential_to_flow(phi)
    mask=torch.zeros(1,9,9);mask[:,2:-2,2:-2]=1
    rho=torch.zeros_like(mask)
    loss=divergence_sink_loss(v,rho,mask,phi=phi,screen_kappa=0,source_density=rho)
    assert loss.item()==pytest.approx(16.) # div(-grad(x²+y²)) = -4
    assert masked_mse(torch.ones(1,2,9,9),torch.zeros(1,2,9,9),mask[:,None]).item()==pytest.approx(1.)


def test_car_anchors_and_empty_sets():
    fg=torch.tensor([[[[-2.,1.,2.,-1.]]]],requires_grad=True)
    clu=torch.tensor([[[2,2,0,1]]]);H=torch.tensor([[[0.,1.,.6,.5]]]);E=1-H
    a=compute_stain_anchored_soft_label_loss(fg,clu,H,alpha=.7,h_threshold=.5)
    b=compute_eosin_anchored_neg_loss(fg,clu,E,H)
    assert a.item()==pytest.approx(torch.nn.functional.binary_cross_entropy_with_logits(fg[0,0,0,1],torch.tensor(.7)).item())
    assert b.item()==pytest.approx(torch.nn.functional.binary_cross_entropy_with_logits(fg[0,0,0,0],torch.tensor(0.)).item())
    assert compute_stain_anchored_soft_label_loss(fg,torch.zeros_like(clu),H).item()==0


def test_scale_jitter_scales_vectors():
    S=32;image=np.zeros((S,S,3),np.uint8);z=np.zeros((S,S),np.float32)
    out=_scale_jitter(image,z,z,z.astype(np.uint8),np.ones((2,S,S),np.float32),z,z,z.astype(np.int32),z,z,1.25,S)
    np.testing.assert_allclose(out[4],1.25)


def test_zero_seed_decoder_drops_all_basins(monkeypatch):
    import chromaflow.decode_variants as dec
    monkeypatch.setattr(dec,'decode_S0',lambda *a:np.ones((16,16),np.uint16))
    fg=np.ones((16,16));flow=np.zeros((2,16,16));center=np.full((16,16),-20.)
    cfg={'seed_threshold':.4,'seed_min_distance':8}
    assert dec.decode_S4(fg,flow,center,cfg,True).max()==0
    assert dec.decode_S4(fg,flow,center,cfg,False).max()==1


def test_aji_unmatched_objects_count_union_once():
    from chromaflow.eval import aji_fast
    gt=np.array([[1,0,2,0],[0,0,0,0]]);pred=np.array([[1,0,0,3],[0,0,0,0]])
    assert aji_fast(gt,pred)==pytest.approx(1/3)


def test_preparation_to_loader_and_model_backward(tmp_path,monkeypatch):
    import chromaflow.prepare as prep
    import chromaflow.protocol as protocol
    data={'train':['sample'],'val':[],'test':[]}
    monkeypatch.setattr(prep,'load_splits',lambda *a:(data,'abc'))
    monkeypatch.setattr(protocol,'load_splits',lambda *a:(data,'abc'))
    base=tmp_path/'data/official_clean/core/monuseg';(base/'images').mkdir(parents=True);(base/'points').mkdir()
    rng=np.random.default_rng(4);rgb=rng.integers(20,240,(250,250,3),dtype=np.uint8)
    Image.fromarray(rgb).save(base/'images/sample.png')
    pm=np.zeros((250,250),np.uint8);pm[10:240:25,10:240:25]=1;np.save(base/'points/sample.npy',pm)
    manifest=prepare(tmp_path,'MO',.05,0)
    saved=json.loads(manifest.read_text());assert len(saved['point_selection']['sample']['retained_yx'])==5
    ds=NucleiV3PatchDataset(tmp_path,'MO','train',point_ratio=.05,point_subset_seed=0,augment=False)
    sample=ds[0];assert sample['image'].shape==(3,224,224)
    batch={k:v.unsqueeze(0) for k,v in sample.items() if isinstance(v,torch.Tensor)}
    model=NISFlowNet(pretrained=False,flow_param='potential');model.eval()
    outputs=model(batch['image']);assert outputs['phi'].shape==(1,1,224,224)
    losses=compute_losses(outputs,batch,lambda_flow=2.,lambda_point=2.,lambda_div=.5,
       screen_kappa=.1,lambda_center=.75,lambda_stain_soft=1.,lambda_eosin_neg=.5)
    total=losses['loss'];assert torch.isfinite(total)
    expected=sum(w*losses[k] for k,w in [('loss_fg',1),('loss_flow',2),('loss_point',2),('loss_div',.5),('loss_center',.75),('loss_stain_soft',1),('loss_eosin_neg',.5)])
    torch.testing.assert_close(total.detach(),expected)
    total.backward()
    assert model.potential_head.weight.grad is not None
    assert torch.isfinite(model.potential_head.weight.grad).all()
    opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=3e-4)
    before=model.potential_head.weight.detach().clone();opt.step()
    assert not torch.equal(before,model.potential_head.weight)
    with pytest.raises(ValueError,match='budget/seed'):NucleiV3PatchDataset(tmp_path,'MO','train',point_ratio=1.)
    with pytest.raises(FileExistsError):prepare(tmp_path,'MO',.05,0)


def test_cellpose_paper_decoder_smoke():
    from chromaflow.decode_variants import decode_S4
    from chromaflow.eval import DEFAULT_INFERENCE_CONFIG
    y,x=np.indices((64,64));fg=np.where((y-32)**2+(x-32)**2<12**2,4.,-4.).astype(np.float32)
    flow=np.stack(((32-x)/250.,(32-y)/250.)).astype(np.float32)
    ce=np.full((64,64),-10.,np.float32);ce[32,32]=10.
    cfg={**DEFAULT_INFERENCE_CONFIG,'device':torch.device('cpu')}
    result=decode_S4(fg,flow,ce,cfg,True)
    assert result.shape==(64,64) and result.max()==1
    assert result[32,32]==1


def test_cli_and_yaml_weights_agree(monkeypatch):
    import sys,yaml
    from chromaflow.train import parse_args
    monkeypatch.setattr(sys,'argv',['train','--dataset','MO'])
    args=parse_args();config=yaml.safe_load(Path('configs/chromaflow_resnet34.yaml').read_text())
    for name,value in config['losses'].items():assert getattr(args,name)==value
    assert args.epochs==150 and args.flow_param=='potential' and args.ckpt_split=='val'
    assert args.stain_aug is False and args.scale_jitter is True


def test_point_displacement_is_fixed_before_tiling():
    from chromaflow.prepare import shift_points
    pm=np.zeros((64,64),np.uint8);pm[10,10]=1;pm[40,40]=1
    out=shift_points(pm,4,3,'sample')
    assert out.sum()==2 and np.array_equal(out,shift_points(pm,4,3,'sample'))
    from scipy.spatial import cKDTree
    dist,_=cKDTree(np.argwhere(pm)).query(np.argwhere(out))
    assert dist.max()<=4


def test_eval_cli_accepts_all_datasets(monkeypatch):
    import sys
    from chromaflow.eval import parse_args
    for name in ['MO','CPM','CONIC','NUINSSEG']:
        monkeypatch.setattr(sys,'argv',['eval','--dataset',name,'--checkpoint','model.pth','--out-dir','run'])
        args=parse_args()
        assert args.dataset==name and args.postprocess=='paper' and args.tta=='hflip,vflip'
