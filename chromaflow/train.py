from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from chromaflow.dataset import NucleiV3PatchDataset
from chromaflow.eval import DEFAULT_INFERENCE_CONFIG, METRIC_NAMES, evaluate_model, write_combined_per_image
from chromaflow.losses import compute_losses
from chromaflow.model import NISFlowNet
from chromaflow.uni_encoder import NISFlowNetUNI
from chromaflow.uni_encoder_hybrid import (
    NISFlowNetUNIHybrid, NISFlowNetUNIHybridLN, NISFlowNetUNIHybridIGA,
    NISFlowNetUNIHybridH, NISFlowNetUNIHybridHPSC2F, NISFlowNetUNIHybridHPSIC,
    NISFlowNetUNIHybridHMTA, NISFlowNetUNIHybridHU2HSRG,
    NISFlowNetUNIHybridHUBIM, NISFlowNetUNIHybridHUCAF, NISFlowNetUNIHybridHUDAC,
    NISFlowNetUNIHybridHUDAR, NISFlowNetUNIHybridHURCF,
    NISFlowNetUNIHybridHURCFV2,
)
from chromaflow.dinov2_encoder import NISFlowNetDINOv2


SNAPSHOT_FILES = ["model.py", "losses.py", "dataset.py", "train.py", "eval.py", "prepare.py", "protocol.py", "decode_variants.py", "tta.py"]


class Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "1", "y"}:
        return True
    if value in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def make_run_dir(root: Path, args: argparse.Namespace) -> Path:
    if args.out_dir:
        out_dir = Path(args.out_dir)
        return out_dir if out_dir.is_absolute() else root / out_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.exp_id or f"{args.method}_{args.dataset}_seed{args.seed}_{timestamp}"
    return root / "runs" / "v3_student" / args.dataset / tag


def snapshot_code(root: Path, run_dir: Path) -> None:
    snapshot_dir = run_dir / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name in SNAPSHOT_FILES:
        path = Path(__file__).resolve().parent / name
        if path.exists():
            shutil.copy2(path, snapshot_dir / name)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    best_val_aji: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "best_val_aji": best_val_aji,
        },
        path,
    )


def load_model_state(path: Path, model: torch.nn.Module, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state") or checkpoint.get("state_dict") or checkpoint
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def append_handoff(root: Path, run_dir: Path, config: dict[str, Any], summaries: dict[str, Any]) -> None:
    path = run_dir / "EXPERIMENT_HANDOFF.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if config.get("lambda_div", 0.0) != 0.0:
        method = config["method"]
    else:
        method = "S0 clean v3" if not config["use_sam_pseudo"] else "v3 + SAM-C"
    flow_desc = (
        "potential-derived flow"
        if config.get("flow_param") == "potential"
        else "direct tanh flow"
    )
    lines = [
        "",
        f"## {now} / {run_dir.name}",
        "",
        "### Goal",
        f"- {method} on {config['dataset']} with seed {config['seed']}.",
        "",
        "### Code",
        "- entry script: `train.py`",
        f"- model: `NISFlowNet` clean R34 v3, {flow_desc}, no SAM module",
        "- inference: v3 Cellpose dynamics, no test-time SAM or GT points",
        f"- run dir: `{run_dir}`",
        "",
        "### Config",
        "```yaml",
        yaml.safe_dump(config, sort_keys=False).strip(),
        "```",
        "",
        "### Results",
        "",
        "| split | AJI | PQ | DQ | SQ | notes |",
        "| ----- | --: | -: | -: | -: | ----- |",
    ]
    for split, summary in summaries.items():
        lines.append(
            f"| {split} | {summary.get('AJI', float('nan')):.4f} | "
            f"{summary.get('PQ', float('nan')):.4f} | {summary.get('DQ', float('nan')):.4f} | "
            f"{summary.get('SQ', float('nan')):.4f} | auto-logged |"
        )
    # E1 PCGS final-stats block (only if pcgs_stats.json exists).
    pcgs_path = run_dir / "pcgs_stats.json"
    if pcgs_path.exists():
        try:
            ps = json.loads(pcgs_path.read_text())
            lines.extend(["", "### PCGS final EM parameters", "```json"])
            ps_compact = {k: v for k, v in ps.items() if k != "history"}
            lines.append(json.dumps(ps_compact, indent=2))
            lines.append("```")
        except Exception:
            pass
    lines.extend(["", "### Decision", "- pending review", ""])
    with path.open("a") as f:
        f.write("\n".join(lines))


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int = 0,
    pcgs_stats: dict | None = None,
    pcgs_em_buffer: dict | None = None,
    spp_state = None,
) -> dict[str, float]:
    model.train()
    # PSIC weight schedule: 0 before warmup_start, ramp to final between start..end, full after.
    lam_psic_final = float(getattr(args, "lambda_psic", 0.0))
    ws = int(getattr(args, "psic_warmup_start", 0))
    we = int(getattr(args, "psic_warmup_end", 0))
    if lam_psic_final <= 0:
        lam_psic_cur = 0.0
    elif epoch < ws:
        lam_psic_cur = 0.0
    elif epoch >= we:
        lam_psic_cur = lam_psic_final
    else:
        lam_psic_cur = lam_psic_final * (epoch - ws) / max(1, we - ws)
    totals = {
        "loss": 0.0,
        "loss_fg": 0.0,
        "loss_flow": 0.0,
        "loss_point": 0.0,
        "loss_div": 0.0,
        "loss_psic": 0.0,
        "loss_sam_fg": 0.0,
        "loss_sam_flow": 0.0,
        "loss_bim": 0.0,
        "loss_hucaf": 0.0,
        "loss_ridge": 0.0,
        "loss_h_boundary": 0.0,
        "loss_stain_dice": 0.0,
        "loss_stain_soft": 0.0,
        "loss_eosin_neg": 0.0, "loss_src": 0.0, "loss_scmp": 0.0,
        "loss_anchor_cdf": 0.0,
        "loss_point_stain": 0.0,
        "loss_he_boundary_fg": 0.0,
        "loss_he_gap_bg": 0.0,
        "loss_aff": 0.0,
        "loss_pcgs": 0.0,
        "loss_spp": 0.0,
        "loss_gap": 0.0,
        "loss_msst": 0.0,
        "loss_niche": 0.0,
        "loss_idg": 0.0,
        "loss_boundary": 0.0,
        "loss_dhs": 0.0,
        "loss_hgfa": 0.0,
        "loss_cpr": 0.0,
        "cpr_pair_acc": 0.0,
        "loss_pef": 0.0,
    }
    count = 0
    for step, batch in enumerate(loader, 1):
        if args.limit_train_batches and step > args.limit_train_batches:
            break
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            # Dual-input models (uni_hybrid_h / uni_hybrid_h_psc2f / uni_hybrid_h_psic) accept dict.
            if args.encoder in ("uni_hybrid_h", "uni_hybrid_h_psc2f", "uni_hybrid_h_psic", "uni_hybrid_h_mta", "uni_hybrid_h_u2h_srg", "uni_hybrid_h_ubim", "uni_hybrid_h_ucaf", "uni_hybrid_h_udac", "uni_hybrid_h_udar", "uni_hybrid_h_urcf", "uni_hybrid_h_urcf_v2", "uni_hybrid_h_lora", "uni_hybrid_h_lora_adapter", "uni_hybrid_h_lora_hca", "uni_hybrid_h_lora_hca_mask", "uni_hybrid_h_lora_hca_v2", "uni_hybrid_h_lora_srma", "uni_hybrid_h_lora_hca_hv", "uni_hybrid_h_lora_sdca", "uni_hybrid_h_lora_hca_r34rgb", "uni_hybrid_h_lora_adasca", "uni_hybrid_h_lora_stainrecon", "uni_hybrid_h_lora_evuc", "uni_hybrid_h_lora_film", "uni_hybrid_h_lora_paa", "uni_hybrid_h_lora_msg", "mssp"):
                model_input = {"image": batch["image"].float(),
                                "image_rgb": batch["image_rgb"].float()}
                # B3 needs instance_id to build per-query mask targets at train time.
                if args.encoder == "uni_hybrid_h_lora_hca_mask" and "instance_id" in batch:
                    model_input["instance_id"] = batch["instance_id"]
                # E1 architecture variants need oracle anchors at train time for
                # per-image stain stats / posterior. At inference time these are
                # absent and the model falls back to default stats / fg_prob.
                if args.encoder in ("uni_hybrid_h_lora_film", "uni_hybrid_h_lora_paa", "uni_hybrid_h_lora_msg"):
                    for k in ("clu", "stain_h", "stain_e", "point_map"):
                        if k in batch:
                            model_input[k] = batch[k]
                outputs = model(model_input)
            else:
                outputs = model(batch["image"].float())
            losses = compute_losses(
                outputs,
                batch,
                lambda_fg=args.lambda_fg,
                lambda_flow=args.lambda_flow,
                lambda_point=args.lambda_point,
                lambda_sam_fg=args.lambda_sam_fg,
                lambda_sam_flow=args.lambda_sam_flow,
                lambda_div=args.lambda_div,
                div_conductance=args.div_conductance,
                screen_kappa=args.screen_kappa,
                sink_sigma=args.sink_sigma,
                sink_adaptive=args.sink_adaptive,
                sink_alpha=args.sink_alpha,
                sink_min=args.sink_min,
                sink_max=args.sink_max,
                stain_alpha=args.stain_alpha,
                lambda_center=args.lambda_center,
                center_sigma=args.center_sigma,
                lambda_aff=args.lambda_aff,
                lambda_mask=args.lambda_mask,
                lambda_mask_iou=args.lambda_mask_iou,
                lambda_h_boundary=args.lambda_h_boundary,
                lambda_stain_dice=args.lambda_stain_dice,
                stain_dice_alpha=args.stain_dice_alpha,
                lambda_stain_soft=args.lambda_stain_soft,
                stain_soft_alpha=args.stain_soft_alpha,
                stain_soft_h_thresh=args.stain_soft_h_thresh,
                lambda_stain_flow=args.lambda_stain_flow,
                stain_flow_h_thresh=args.stain_flow_h_thresh,
                stain_flow_max_dist=args.stain_flow_max_dist,
                stain_flow_adaptive=args.stain_flow_adaptive,
                stain_flow_adapt_alpha=args.stain_flow_adapt_alpha,
                stain_flow_r_min=args.stain_flow_r_min,
                stain_flow_r_max=args.stain_flow_r_max,
                stain_soft_max_dist=args.stain_soft_max_dist,
                stain_soft_target_mode=args.stain_soft_target_mode,
                stain_soft_proximity_kernel=args.stain_soft_proximity_kernel,
                stain_soft_trigger_opening=args.stain_soft_trigger_opening,
                expand_cluster_via_stain=args.expand_cluster_via_stain,
                expand_cluster_h_thresh=args.expand_cluster_h_thresh,
                expand_cluster_max_dist=args.expand_cluster_max_dist,
                lambda_eosin_neg=args.lambda_eosin_neg,
                eosin_neg_tau_e=args.eosin_neg_tau_e,
                eosin_neg_tau_h=args.eosin_neg_tau_h,
                eosin_neg_max_dist=args.eosin_neg_max_dist,
                eosin_neg_min_dist=args.eosin_neg_min_dist,
                eosin_neg_fg_gate=args.eosin_neg_fg_gate,
                lambda_src=args.lambda_src,
                src_tau_e=args.src_tau_e,
                src_tau_h=args.src_tau_h,
                src_max_dist=args.src_max_dist,
                src_margin=args.src_margin,
                lambda_anchor_cdf=args.lambda_anchor_cdf,
                anchor_cdf_n_bins=args.anchor_cdf_n_bins,
                anchor_cdf_smooth_eps=args.anchor_cdf_smooth_eps,
                anchor_cdf_use_e=args.anchor_cdf_use_e,
                anchor_cdf_max_dist=args.anchor_cdf_max_dist,
                anchor_cdf_target_clip=args.anchor_cdf_target_clip,
                lambda_point_stain=getattr(args, "lambda_point_stain", 0.0),
                point_stain_mode=getattr(args, "point_stain_mode", "posterior"),
                point_stain_adaptive_radius=getattr(args, "point_stain_adaptive_radius", True),
                point_stain_bg_min_dist=getattr(args, "point_stain_bg_min_dist", 45.0),
                lambda_he_boundary_fg=args.lambda_he_boundary_fg,
                he_boundary_tau_h=args.he_boundary_tau_h,
                he_boundary_tau_e=args.he_boundary_tau_e,
                lambda_he_gap_bg=args.lambda_he_gap_bg,
                he_gap_tau_h=args.he_gap_tau_h,
                he_gap_tau_e=args.he_gap_tau_e,
                he_quad_max_dist=args.he_quad_max_dist,
                adaptive_proximity=args.adaptive_proximity,
                adapt_alpha=args.adapt_alpha,
                adapt_r_min=args.adapt_r_min,
                adapt_r_max=args.adapt_r_max,
                lambda_scmp=args.lambda_scmp,
                lambda_stain_recon=args.lambda_stain_recon,
                use_cpgs=args.use_cpgs, cpgs_k=args.cpgs_k, cpgs_alpha=args.cpgs_alpha,
                lambda_evidential_mse=args.lambda_evidential_mse,
                lambda_evidential_kl=args.lambda_evidential_kl,
                uncertainty_gate=args.uncertainty_gate,
                uncertainty_gate_pow=args.uncertainty_gate_pow,
                scmp_sharpness=args.scmp_sharpness,
                scmp_max_dist=args.scmp_max_dist,
                scmp_confidence_floor=args.scmp_confidence_floor,
                lambda_hv=args.lambda_hv,
                hv_stain_alpha=args.hv_stain_alpha,
                aff_radius=args.aff_radius,
                lambda_psic=lam_psic_cur,
                psic_r_pos=getattr(args, "psic_r_pos", 10.0),
                psic_beta=getattr(args, "psic_beta", 0.6),
                psic_r_neg=getattr(args, "psic_r_neg", 64.0),
                psic_k_min=getattr(args, "psic_k_min", 4),
                psic_k_anchor=getattr(args, "psic_k_anchor", 8),
                psic_tau=getattr(args, "psic_tau", 0.1),
                psic_adaptive_radius=getattr(args, "psic_adaptive_radius", False),
                psic_r_pos_max=getattr(args, "psic_r_pos_max", 8.0),
                psic_alpha_dnn=getattr(args, "psic_alpha_dnn", 0.25),
                psic_top_k_neg=getattr(args, "psic_top_k_neg", 0),
                psic_use_margin=getattr(args, "psic_use_margin", False),
                psic_margin_value=getattr(args, "psic_margin_value", 0.15),
                psic_neg_weight=getattr(args, "psic_neg_weight", 0.5),
                lambda_bim=getattr(args, "lambda_bim", 0.0),
                bim_tau=getattr(args, "bim_tau", 0.3),
                bim_r_max=getattr(args, "bim_r_max", 15.0),
                lambda_hucaf=getattr(args, "lambda_hucaf", 0.0),
                lambda_ridge=getattr(args, "lambda_ridge", 0.0),
                ridge_sigma_b=getattr(args, "ridge_sigma_b", 5.0),
                ridge_r_core=getattr(args, "ridge_r_core", 0.0),
                use_scs=getattr(args, "use_scs", False),
                scs_min_weight=getattr(args, "scs_min_weight", 0.2),
                # E1 PCGS
                use_pcgs=getattr(args, "use_pcgs", False),
                pcgs_stats=pcgs_stats,
                pcgs_lambda=getattr(args, "pcgs_lambda", 1.0),
                pcgs_max_dist=getattr(args, "pcgs_max_dist", 30.0),
                pcgs_mode=getattr(args, "pcgs_mode", "anchor"),
                pcgs_bg_anchor_min_dist=getattr(args, "pcgs_bg_anchor_min_dist", 40.0),
                pcgs_prior_fg=getattr(args, "pcgs_prior_fg", 0.5),
                pcgs_additive=getattr(args, "pcgs_additive", False),
                pcgs_target_temperature=getattr(args, "pcgs_target_temperature", 1.0),
                pcgs_warmup_active=(
                    getattr(args, "use_pcgs", False)
                    and getattr(args, "pcgs_mode", "anchor") == "em"
                    and epoch < int(getattr(args, "pcgs_em_warmup", 9999))
                ),
                pcgs_return_em=(
                    getattr(args, "use_pcgs", False)
                    and getattr(args, "pcgs_mode", "anchor") == "em"
                    and epoch >= int(getattr(args, "pcgs_em_warmup", 9999))
                    and pcgs_em_buffer is not None
                ),
                # SPP
                use_spp=getattr(args, "use_spp", False),
                spp_state=spp_state,
                lambda_spp=float(getattr(args, "lambda_spp", 0.0)),
                spp_gamma=float(getattr(args, "spp_gamma", 1.0)),
                # GAP
                use_gap=getattr(args, "use_gap", False),
                lambda_gap=float(getattr(args, "lambda_gap", 0.0)),
                gap_R=float(getattr(args, "gap_R", 7.5)),
                gap_tau=float(getattr(args, "gap_tau", 3.5)),
                gap_mode=str(getattr(args, "gap_mode", "soft")),
                gap_d_fg_max=float(getattr(args, "gap_d_fg_max", 4.0)),
                gap_d_bg_min=float(getattr(args, "gap_d_bg_min", 12.0)),
                gap_target_fg=float(getattr(args, "gap_target_fg", 0.7)),
                gap_target_bg=float(getattr(args, "gap_target_bg", 0.0)),
                # MSST
                use_msst=getattr(args, "use_msst", False),
                lambda_msst=float(getattr(args, "lambda_msst", 0.0)),
                msst_d_fg_max=float(getattr(args, "msst_d_fg_max", 2.0)),
                msst_d_bg_min=float(getattr(args, "msst_d_bg_min", 10.0)),
                msst_h_kernel=int(getattr(args, "msst_h_kernel", 5)),
                msst_h_fg_min=float(getattr(args, "msst_h_fg_min", 0.5)),
                msst_h_bg_max=float(getattr(args, "msst_h_bg_max", 0.3)),
                msst_target_fg=float(getattr(args, "msst_target_fg", 0.7)),
                msst_target_bg=float(getattr(args, "msst_target_bg", 0.0)),
                # NICHE
                use_niche=getattr(args, "use_niche", False),
                lambda_niche=float(getattr(args, "lambda_niche", 0.0)),
                niche_rho_max=float(getattr(args, "niche_rho_max", 0.2)),
                niche_d_fg_max=float(getattr(args, "niche_d_fg_max", 2.0)),
                niche_d_bg_min=float(getattr(args, "niche_d_bg_min", 12.0)),
                niche_target_fg=float(getattr(args, "niche_target_fg", 0.7)),
                niche_target_bg=float(getattr(args, "niche_target_bg", 0.0)),
                # IDG
                use_idg=getattr(args, "use_idg", False),
                lambda_idg=float(getattr(args, "lambda_idg", 0.0)),
                idg_sigma=float(getattr(args, "idg_sigma", 3.0)),
                idg_h_alpha=float(getattr(args, "idg_h_alpha", 2.0)),
                idg_trigger_q=float(getattr(args, "idg_trigger_q", 0.95)),
                idg_target=float(getattr(args, "idg_target", 0.0)),
                use_boundary=getattr(args, "use_boundary", False),
                lambda_boundary=float(getattr(args, "lambda_boundary", 0.0)),
                boundary_sigma=float(getattr(args, "boundary_sigma", 3.0)),
                use_boundary_h=getattr(args, "use_boundary_h", False),
                boundary_h_sigma=float(getattr(args, "boundary_h_sigma", 1.5)),
                boundary_h_floor=float(getattr(args, "boundary_h_floor", 0.0)),
                use_csbb=getattr(args, "use_csbb", False),
                csbb_rb=float(getattr(args, "csbb_rb", 18.0)),
                csbb_h_floor=float(getattr(args, "csbb_h_floor", 0.4)),
                use_gcsbb=getattr(args, "use_gcsbb", False),
                gcsbb_fg_tau=float(getattr(args, "gcsbb_fg_tau", 0.3)),
                gcsbb_crowd_r=float(getattr(args, "gcsbb_crowd_r", 18.0)),
                # CGFS
                use_dhs=getattr(args, "use_dhs", False),
                lambda_dhs=float(getattr(args, "lambda_dhs", 0.0)),
                dhs_sigma=float(getattr(args, "dhs_sigma", 1.5)),
                dhs_target_low=float(getattr(args, "dhs_target_low", 0.0)),
                dhs_target_high=float(getattr(args, "dhs_target_high", 1.0)),
                use_hgfa=getattr(args, "use_hgfa", False),
                lambda_hgfa=float(getattr(args, "lambda_hgfa", 0.0)),
                hgfa_sigma=float(getattr(args, "hgfa_sigma", 1.5)),
                hgfa_d_max=float(getattr(args, "hgfa_d_max", 5.0)),
                hgfa_grad_floor=float(getattr(args, "hgfa_grad_floor", 0.0)),
                hgfa_weight_pow=float(getattr(args, "hgfa_weight_pow", 1.0)),
                use_cpr=getattr(args, "use_cpr", False),
                lambda_cpr=float(getattr(args, "lambda_cpr", 0.0)),
                cpr_n_pairs=int(getattr(args, "cpr_n_pairs", 64)),
                cpr_rho_diff_min=float(getattr(args, "cpr_rho_diff_min", 0.3)),
                cpr_margin_init=float(getattr(args, "cpr_margin_init", 0.05)),
                cpr_margin_final=float(getattr(args, "cpr_margin_final", 0.25)),
                cpr_margin_warmup_epochs=int(getattr(args, "cpr_margin_warmup_epochs", 40)),
                current_epoch=int(epoch),
                use_pef=getattr(args, "use_pef", False),
                lambda_pef=float(getattr(args, "lambda_pef", 0.0)),
            )
            if getattr(args, "use_pace", False) and getattr(args, "encoder", "") == "resnet34":
                from chromaflow.pace_loss import compute_pace_loss, hed_jitter_tensor
                _img = batch["image"].float()
                _vb = hed_jitter_tensor(_img, sigma=float(getattr(args, "pace_jit_sigma", 0.08)))
                _pa = model.forward_pace(_img)
                _pb = model.forward_pace(_vb)
                loss_pace = compute_pace_loss(_pa, _pb, batch["point_map"], batch["stain_h"],
                                              tau=float(getattr(args, "pace_tau", 0.2)),
                                              r_n=float(getattr(args, "pace_rn", 48.0)),
                                              eta=float(getattr(args, "pace_eta", 0.0)),
                                              use_he=getattr(args, "pace_he", False))
                losses["loss"] = losses["loss"] + float(getattr(args, "lambda_pace", 0.05)) * loss_pace
                losses["loss_pace"] = loss_pace.detach()
        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        # E1 PCGS EM accumulation (post-warmup only): use detached sigmoid of
        # the current model's fg head as the soft responsibility r(x). Sufficient
        # statistics accumulated across batches are reduced into M-step updates
        # at epoch end (see main()).
        em_in = losses.pop("_pcgs_em_inputs", None) if isinstance(losses, dict) else None
        if (em_in is not None
                and pcgs_em_buffer is not None
                and getattr(args, "use_pcgs", False)
                and epoch >= int(getattr(args, "pcgs_em_warmup", 9999))):
            from chromaflow.losses import pcgs_em_accumulate as _pcgs_acc
            with torch.no_grad():
                fg_sig = torch.sigmoid(outputs["fg_logits"].detach())
                acc = _pcgs_acc(fg_sig, em_in)
                for k, v in acc.items():
                    pcgs_em_buffer[k] = pcgs_em_buffer.get(k, 0.0) + v

        batch_size = int(batch["image"].shape[0])
        count += batch_size
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * batch_size
        if step % args.log_freq == 0:
            msg = " ".join(f"{key}={totals[key] / max(count, 1):.4f}" for key in totals)
            print(f"[train] step={step}/{len(loader)} {msg}", flush=True)
    return {key: totals[key] / max(count, 1) for key in totals}


def write_epoch_log(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        fieldnames = ["epoch", "val_AJI"] + [
            "loss",
            "loss_fg",
            "loss_flow",
            "loss_point",
            "loss_div",
            "loss_psic",
            "loss_sam_fg",
            "loss_sam_flow",
            "loss_bim",
            "loss_hucaf",
            "loss_ridge",
            "loss_pcgs",
            "loss_spp",
            "loss_gap",
            "loss_msst",
            "loss_niche",
            "loss_idg",
            "loss_boundary",
            "loss_dhs",
            "loss_hgfa",
            "loss_anchor_cdf",
            "loss_point_stain",
            "loss_he_boundary_fg",
            "loss_he_gap_bg",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_val_instance_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = (
        ["epoch"]
        + [f"val_{name}" for name in METRIC_NAMES]
        + ["val_loss", "val_fg_acc"]
        + [
            "train_loss",
            "train_loss_fg",
            "train_loss_flow",
            "train_loss_point",
            "train_loss_div",
            "train_loss_sam_fg",
            "train_loss_sam_flow",
        ]
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dataset", choices=["MO", "CPM", "CONIC", "NUINSSEG"], required=True)
    parser.add_argument("--sags-snap-window", type=int, default=0,
                        help="D9 SAGS: window radius for snapping clicks to local H peaks (0=off).")
    parser.add_argument("--clu-subdir", default="labels_clu",
                        help="Cluster-label subdirectory; labels_clu is the manuscript path. Other folders are exploratory variants.")
    parser.add_argument("--method", default="v3")
    parser.add_argument("--exp-id", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lambda-fg", type=float, default=1.0)
    parser.add_argument("--lambda-flow", type=float, default=2.0)
    parser.add_argument("--lambda-point", type=float, default=2.0)
    parser.add_argument("--flow-param", choices=["direct", "potential", "poisson"], default="potential")
    parser.add_argument("--encoder", choices=[
        "resnet34", "resnet18", "resnet50", "resnet101", "sacaf", "r34dual", "latef", "uni", "uni_hybrid", "uni_hybrid_ln", "uni_hybrid_iga",
        "uni_hybrid_h", "uni_hybrid_h_psc2f", "uni_hybrid_h_psic", "uni_hybrid_h_mta",
        "uni_hybrid_h_u2h_srg", "uni_hybrid_h_ubim", "uni_hybrid_h_ucaf", "uni_hybrid_h_udac",
        "uni_hybrid_h_udar", "uni_hybrid_h_urcf", "uni_hybrid_h_urcf_v2", "uni_hybrid_h_lora", "uni_hybrid_h_lora_adapter", "uni_hybrid_h_lora_hca", "uni_hybrid_h_lora_hca_mask", "uni_hybrid_h_lora_hca_v2", "uni_hybrid_h_lora_srma", "uni_hybrid_h_lora_hca_hv", "uni_hybrid_h_lora_sdca", "uni_hybrid_h_lora_hca_r34rgb", "uni_hybrid_h_lora_adasca", "uni_hybrid_h_lora_stainrecon", "uni_hybrid_h_lora_evuc", "uni_hybrid_h_lora_film", "uni_hybrid_h_lora_paa", "uni_hybrid_h_lora_msg", "mssp", "dinov2", "scde",
    ], default="resnet34")
    parser.add_argument("--uni-path", default="weights/uni/UNI/pytorch_model.bin")
    parser.add_argument("--lambda-div", type=float, default=0.5)
    parser.add_argument("--div-conductance", choices=["constant", "stain", "eosin"], default="constant")
    parser.add_argument("--screen-kappa", type=float, default=0.1,
                        help="U1 screened-Poisson: (Delta-kappa^2)phi=-rho; residual += kappa^2*phi. Needs flow_param=potential. 0=plain Poisson.")
    parser.add_argument("--sink-sigma", type=float, default=3.0,
                         help="L_div: fixed sink sigma (used when --no-sink-adaptive)")
    parser.add_argument("--sink-adaptive", action=argparse.BooleanOptionalAction, default=False,
                         help="L_div: enable per-point density-adaptive sink: sigma_i = clip(alpha*d_NN, min, max)")
    parser.add_argument("--sink-alpha", type=float, default=0.25,
                         help="L_div: adaptive sink alpha (scaling factor on d_NN)")
    parser.add_argument("--sink-min", type=float, default=3.0,
                         help="L_div: adaptive sink minimum sigma")
    parser.add_argument("--sink-max", type=float, default=8.0,
                         help="L_div: adaptive sink maximum sigma")
    parser.add_argument("--stain-alpha", type=float, default=0.1)
    parser.add_argument("--lambda-center", type=float, default=0.75)
    parser.add_argument("--center-sigma", type=float, default=3.0)
    parser.add_argument("--lambda-aff", type=float, default=0.0)
    parser.add_argument("--lambda-mask", type=float, default=0.0,
                        help="B3 instance mask decoder loss weight (BCE+Dice).")
    parser.add_argument("--lambda-mask-iou", type=float, default=0.0,
                        help="B3 IoU prediction head loss weight.")
    parser.add_argument("--lambda-h-boundary", type=float, default=0.0,
                        help="B6 hematoxylin-gradient boundary loss weight.")
    parser.add_argument("--lambda-stain-dice", type=float, default=0.0,
                        help="B7 SCR-interior stain-weighted Dice loss weight.")
    parser.add_argument("--stain-dice-alpha", type=float, default=0.5,
                        help="B7 stain-weighting strength: weight = 1+alpha*norm(stain_h).")
    parser.add_argument("--lambda-stain-flow", type=float, default=0.0)
    parser.add_argument("--stain-flow-h-thresh", type=float, default=0.5)
    parser.add_argument("--stain-flow-max-dist", type=int, default=30)
    parser.add_argument("--stain-flow-adaptive", action="store_true")
    parser.add_argument("--stain-flow-adapt-alpha", type=float, default=0.6)
    parser.add_argument("--stain-flow-r-min", type=int, default=8)
    parser.add_argument("--stain-flow-r-max", type=int, default=30)
    parser.add_argument("--lambda-stain-soft", type=float, default=1.0,
                        help="B9 SCR-soft: soft fg supervision in clu==2 high-H zone.")
    parser.add_argument("--stain-soft-alpha", type=float, default=0.7,
                        help="B9 soft fg target value (0-1).")
    parser.add_argument("--stain-soft-max-dist", type=int, default=0,
                        help="B13 proximity gate radius (px) for L_stain_soft. 0=disabled.")
    parser.add_argument("--stain-soft-target-mode", choices=["binary", "continuous"], default="binary",
                        help="B14c: continuous H-weighted target vs binary threshold.")
    parser.add_argument("--stain-soft-proximity-kernel", choices=["square", "disk"], default="square",
                        help="B14a: disk-shaped proximity (Euclidean) vs square (max_pool).")
    parser.add_argument("--stain-soft-trigger-opening", type=int, default=0,
                        help="C1: morphological-opening radius (px) on stain_soft trigger mask. 0=disabled.")
    parser.add_argument("--expand-cluster-via-stain", action="store_true",
                        help="B15: promote stain-soft pixels to HARD fg supervision in L_fg.")
    parser.add_argument("--expand-cluster-h-thresh", type=float, default=0.5,
                        help="B15: min H_norm to qualify for cluster expansion.")
    parser.add_argument("--expand-cluster-max-dist", type=int, default=15,
                        help="B15: proximity gate radius (px) for cluster expansion.")
    parser.add_argument("--eosin-neg-max-dist", type=int, default=0,
                        help="B13 proximity gate radius (px) for L_eosin_neg. 0=disabled.")
    parser.add_argument("--eosin-neg-min-dist", type=int, default=0,
                        help="B16 inner-radius gate (px) for L_eosin_neg. Pixels within min_dist of any GT point are EXCLUDED. 0=disabled.")
    parser.add_argument("--eosin-neg-fg-gate", type=float, default=0.0,
                        help="C2 self-confidence gate: drop hard-bg pixels where sigmoid(fg_logits) >= gate. 0=disabled. Typical 0.5.")
    # L_src: eosin-anchored source-ridge on flow divergence (same gate as E-).
    parser.add_argument("--lambda-src", type=float, default=0.0,
                        help="L_src: weight for eosin-anchored positive-divergence (source-ridge) loss. 0=disabled.")
    parser.add_argument("--src-tau-e", type=float, default=0.5,
                        help="L_src: min normalized E to qualify as cytoplasm gap (matches eosin_neg).")
    parser.add_argument("--src-tau-h", type=float, default=0.5,
                        help="L_src: max normalized H to qualify as cytoplasm gap (matches eosin_neg).")
    parser.add_argument("--src-max-dist", type=int, default=30,
                        help="L_src: proximity gate radius (px) around GT points. 0=disabled.")
    parser.add_argument("--src-margin", type=float, default=0.3,
                        help="L_src: hinge target; penalize div(flow) below this on the gate (push positive source ridge).")
    parser.add_argument("--adaptive-proximity", action="store_true",
                        help="C7: per-image adaptive proximity radius based on median NN distance.")
    parser.add_argument("--adapt-alpha", type=float, default=0.6,
                        help="C7: scale factor on median(NN_dist). radius = clip(alpha * median_NN, r_min, r_max).")
    parser.add_argument("--adapt-r-min", type=int, default=15,
                        help="C7: minimum adaptive radius (px).")
    parser.add_argument("--adapt-r-max", type=int, default=50,
                        help="C7: maximum adaptive radius (px).")
    parser.add_argument("--use-cpgs", action="store_true",
                        help="C4 CPGS: chromophore-polarization-gated supervision (replaces hard H/E thresholds).")
    parser.add_argument("--cpgs-k", type=float, default=4.0,
                        help="C4 sigmoid sharpness on polarization.")
    parser.add_argument("--cpgs-alpha", type=float, default=1.0,
                        help="C4 power on polarization weight.")
    parser.add_argument("--lambda-evidential-mse", type=float, default=0.0,
                        help="D8 evidential MSE loss weight.")
    parser.add_argument("--lambda-evidential-kl", type=float, default=0.0,
                        help="D8 evidential KL regularization weight (anneal from 0).")
    parser.add_argument("--uncertainty-gate", type=float, default=0.0,
                        help="D8 multiplier for uncertainty gating on DS-MPS (>0 enables).")
    parser.add_argument("--uncertainty-gate-pow", type=float, default=1.0,
                        help="D8 power on uncertainty before gating.")
    parser.add_argument("--lambda-stain-recon", type=float, default=0.0,
                        help="D7 stain reconstruction aux loss weight.")
    parser.add_argument("--lambda-scmp", type=float, default=0.0,
                        help="B17 SCMP: unified Stain-Confidence Modality Prior loss weight.")
    parser.add_argument("--scmp-sharpness", type=float, default=8.0,
                        help="B17 SCMP sigmoid sharpness k in sigmoid(k*(H-E)).")
    parser.add_argument("--scmp-max-dist", type=int, default=30,
                        help="B17 SCMP proximity gate (px). 0 disables gate.")
    parser.add_argument("--scmp-confidence-floor", type=float, default=1.0,
                        help="B18 SCMP per-pixel weight floor in [0,1]. 1.0=uniform (B17), 0.0=pure |diff| weight.")
    parser.add_argument("--stain-soft-h-thresh", type=float, default=0.5,
                        help="B9 norm(stain_h) threshold for soft-supervised pixels.")
    parser.add_argument("--lambda-eosin-neg", type=float, default=0.5,
                        help="B12 MPS-negative: eosin-anchored hard-bg loss in clu==2 high-E low-H zone.")
    parser.add_argument("--eosin-neg-tau-e", type=float, default=0.5,
                        help="B12 norm(stain_e) threshold (above qualifies as cytoplasm).")
    parser.add_argument("--eosin-neg-tau-h", type=float, default=0.5,
                        help="B12 norm(stain_h) threshold (below qualifies as not-nucleus).")
    # E5 DCSS: Direction-Corrected Stain Supervision. Replaces L_eosin_neg with
    # two correctly-directed H/E quadrant supervisions.
    parser.add_argument("--lambda-he-boundary-fg", type=float, default=0.0,
                        help="E5 DCSS-boundary-fg: clu==2 high-H high-E -> target=1 (85%% fg per oracle).")
    parser.add_argument("--he-boundary-tau-h", type=float, default=0.5,
                        help="E5 H_norm lower bound for boundary-fg trigger.")
    parser.add_argument("--he-boundary-tau-e", type=float, default=0.5,
                        help="E5 E_norm lower bound for boundary-fg trigger.")
    parser.add_argument("--lambda-he-gap-bg", type=float, default=0.0,
                        help="E5 DCSS-gap-bg: clu==2 low-H low-E -> target=0 (99%% bg per oracle).")
    parser.add_argument("--he-gap-tau-h", type=float, default=0.5,
                        help="E5 H_norm upper bound for gap-bg trigger.")
    parser.add_argument("--he-gap-tau-e", type=float, default=0.15,
                        help="E5 E_norm upper bound for gap-bg trigger.")
    parser.add_argument("--he-quad-max-dist", type=float, default=30.0,
                        help="E5 proximity gate radius (px) for both DCSS losses.")
    # E4 Anchor-CDF: non-parametric per-image empirical Bayesian posterior
    # supervision on BoNuS ignore zone (clu==2). Replaces L_stain_soft/L_eosin_neg
    # hand-crafted thresholds with adaptive Bayesian formulation.
    parser.add_argument("--lambda-anchor-cdf", type=float, default=0.0,
                        help="E4 Anchor-CDF loss weight. 0=disabled.")
    parser.add_argument("--anchor-cdf-n-bins", type=int, default=64,
                        help="E4 histogram resolution for empirical CDF estimation.")
    parser.add_argument("--anchor-cdf-smooth-eps", type=float, default=1.0,
                        help="E4 Laplace smoothing pseudo-count per bin.")
    parser.add_argument("--no-anchor-cdf-use-e", dest="anchor_cdf_use_e",
                        action="store_false",
                        help="E4 disable E-channel posterior combination (use H-only).")
    parser.set_defaults(anchor_cdf_use_e=True)
    parser.add_argument("--anchor-cdf-max-dist", type=float, default=30.0,
                        help="E4 proximity gate radius (px). 0=disabled.")
    parser.add_argument("--anchor-cdf-target-clip", type=float, default=0.05,
                        help="E4 target clip ε: clamp soft BCE target to [ε, 1-ε].")
    # PSPS: Point-Stain Posterior Supervision. BoNuS-free replacement candidate
    # for L_stain_soft/L_eosin_neg/PCGS; uses only H/E channels and point_map.
    parser.add_argument("--lambda-point-stain", type=float, default=0.0,
                        help="PSPS loss weight. 0=disabled.")
    parser.add_argument("--point-stain-mode",
                        choices=["posterior", "h_only", "e_only", "no_distance", "shuffled_he"],
                        default="posterior",
                        help="PSPS target builder mode / mechanism ablation.")
    parser.add_argument("--point-stain-adaptive-radius",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="PSPS adaptive radii from median nearest-neighbor point distance.")
    parser.add_argument("--point-stain-bg-min-dist", type=float, default=45.0,
                        help="PSPS minimum far-background anchor distance in pixels.")
    parser.add_argument("--lambda-hv", type=float, default=0.0,
                        help="B5 HSDR hv-map regression loss weight.")
    parser.add_argument("--hv-stain-alpha", type=float, default=1.0,
                        help="B5 HSDR stain weighting strength (1+a*stain_h).")
    parser.add_argument("--aff-radius", type=float, default=5.0)
    # PSIC (Point-Sink Instance Contrastive) — HUF-Net-V1
    parser.add_argument("--lambda-psic", type=float, default=0.0,
                        help="Final PSIC weight after warmup. Recommended 0.03.")
    parser.add_argument("--psic-warmup-start", type=int, default=60,
                        help="Epoch to start ramping PSIC weight from 0.")
    parser.add_argument("--psic-warmup-end", type=int, default=80,
                        help="Epoch to reach full PSIC weight.")
    parser.add_argument("--psic-r-pos", type=float, default=10.0)
    parser.add_argument("--psic-beta", type=float, default=0.6)
    parser.add_argument("--psic-r-neg", type=float, default=64.0)
    parser.add_argument("--psic-k-min", type=int, default=4)
    parser.add_argument("--psic-k-anchor", type=int, default=8)
    parser.add_argument("--psic-tau", type=float, default=0.1)
    # PSIC-v2 additions
    parser.add_argument("--psic-adaptive-radius", action="store_true",
                        help="v2A/B/C: per-nucleus radius = min(r_max, alpha * d_NN)")
    parser.add_argument("--psic-r-pos-max", type=float, default=8.0)
    parser.add_argument("--psic-alpha-dnn", type=float, default=0.25)
    parser.add_argument("--psic-top-k-neg", type=int, default=0,
                        help="v2B/C: keep top-K hard negatives by similarity. 0 = all neighbors.")
    parser.add_argument("--psic-use-margin", action="store_true",
                        help="v2C: margin contrast (replaces InfoNCE)")
    parser.add_argument("--psic-margin-value", type=float, default=0.15)
    parser.add_argument("--psic-neg-weight", type=float, default=0.5)
    # MTA (Morphology-guided Token Adapter) flags
    parser.add_argument("--mta-hidden-ch", type=int, default=256,
                         help="MTA: hidden channels in FiLM generator")
    parser.add_argument("--mta-detach-h", type=str2bool, nargs="?", const=True, default=True,
                         help="MTA: stop-gradient on H feature before FiLM gen (recommended True for v1)")
    parser.add_argument("--mta-init-scale", type=float, default=0.1,
                         help="MTA: initial value of learnable modulation strength eta")
    # U2H-SRG (UNI-to-H Semantic Reliability Gate)
    parser.add_argument("--u2h-alpha", type=float, default=0.5,
                         help="U2H-SRG: fixed modulation strength alpha")
    parser.add_argument("--u2h-hidden-ch", type=int, default=128,
                         help="U2H-SRG: hidden channels in gate generator")
    parser.add_argument("--u2h-detach-uni", type=str2bool, nargs="?", const=True, default=True,
                         help="U2H-SRG: stop-gradient through UNI cond. (recommended True)")
    parser.add_argument("--u2h-no-wd", type=str2bool, nargs="?", const=True, default=True,
                         help="U2H-SRG: exclude gate params from weight decay (separate optimizer group)")
    # HU-BIM (H-UNI Boundary Inference Module)
    parser.add_argument("--lambda-bim", type=float, default=0.0,
                         help="HU-BIM: weight for auxiliary boundary BCE loss")
    parser.add_argument("--bim-tau", type=float, default=0.3,
                         help="HU-BIM: ridge half-width fraction of d_1")
    parser.add_argument("--bim-r-max", type=float, default=15.0,
                         help="HU-BIM: cap on d_1 for thresh (full-res px)")
    parser.add_argument("--bim-mid-ch", type=int, default=64,
                         help="HU-BIM: middle channels in boundary fuse block")
    # HU-CAF (Cross-branch Aux Foreground)
    parser.add_argument("--lambda-hucaf", type=float, default=0.0,
                         help="HU-CAF: weight for cross-branch fg distill (MSE on sigmoid)")
    # HU-DAC (Disagreement-Aware Channel gating)
    parser.add_argument("--hudac-alpha", type=float, default=0.1,
                         help="HU-DAC: channel gate strength alpha (bounded modulation)")
    parser.add_argument("--hudac-hidden", type=int, default=256,
                         help="HU-DAC: hidden width in gate MLP")
    # RCF-Head (Ridge-Conditioned Flow Head)
    parser.add_argument("--lambda-ridge", type=float, default=0.0,
                         help="RCF-Head: weight for SmoothL1 ridge loss")
    parser.add_argument("--ridge-sigma-b", type=float, default=5.0,
                         help="RCF-Head: soft ridge target Gaussian sigma at 1/8 resolution")
    parser.add_argument("--ridge-ch", type=int, default=64,
                         help="RCF-Head: ridge feature channels")
    parser.add_argument("--ridge-r-core", type=float, default=0.0,
                         help="RCF-v2: exclude ridge target where d_1 <= r_core (in 1/8 px)")
    parser.add_argument("--ridge-residual-scale", type=float, default=0.05,
                         help="RCF-v2: residual scale lambda_res for flow correction")
    # SCS: Stain-Confidence Sampling on clu==1
    parser.add_argument("--use-scs", action="store_true",
                         help="SCS: per-pixel chromophore-polarization weight on L_fg/L_flow within cluster disk.")
    parser.add_argument("--scs-min-weight", type=float, default=0.2,
                         help="SCS: minimum weight clamp for E-dominant cluster-disk pixels.")
    # SPP (Stain Polarization Posterior) — Module 1 of chromophore paper
    parser.add_argument("--use-spp", action="store_true",
                         help="SPP: enable Stain Polarization Posterior loss on clu==2.")
    parser.add_argument("--spp-k-init", type=float, default=2.0,
                         help="SPP: initial polarization slope (EM-calibrated).")
    parser.add_argument("--spp-b-init", type=float, default=-0.5,
                         help="SPP: initial prior bias.")
    parser.add_argument("--spp-gamma", type=float, default=1.0,
                         help="SPP: confidence weight exponent |ρ|^γ.")
    parser.add_argument("--lambda-spp", type=float, default=0.5,
                         help="SPP: loss weight (effective when --use-spp).")
    parser.add_argument("--spp-ema-momentum", type=float, default=0.5,
                         help="SPP: EMA momentum on (k, b) updates (1.0 = replace each epoch).")
    parser.add_argument("--spp-update-freq", type=int, default=1,
                         help="SPP: refit (k, b) every N epochs (default 1).")
    # GAP (Geometric Anchor-Proximity Posterior)
    parser.add_argument("--use-gap", action="store_true",
                         help="GAP: enable geometric distance-to-clu1 posterior loss on clu==2.")
    parser.add_argument("--lambda-gap", type=float, default=0.5,
                         help="GAP: loss weight (effective when --use-gap).")
    parser.add_argument("--gap-R", type=float, default=7.5,
                         help="GAP: posterior decay center R (px). Default = REAL GT MO calibration.")
    parser.add_argument("--gap-tau", type=float, default=3.5,
                         help="GAP: posterior softness τ (px).")
    parser.add_argument("--gap-mode", choices=["soft", "sparsehard"], default="soft",
                         help="GAP: 'soft' = dense σ((R-d)/τ) target on all clu==2; 'sparsehard' = V3-style sparse with d<d_fg_max → target_fg, d>d_bg_min → target_bg.")
    parser.add_argument("--gap-d-fg-max", type=float, default=4.0,
                         help="GAP sparsehard mode: max d (px) to trigger fg supervision.")
    parser.add_argument("--gap-d-bg-min", type=float, default=12.0,
                         help="GAP sparsehard mode: min d (px) to trigger bg supervision.")
    parser.add_argument("--gap-target-fg", type=float, default=0.7,
                         help="GAP sparsehard mode: fg supervision target value.")
    parser.add_argument("--gap-target-bg", type=float, default=0.0,
                         help="GAP sparsehard mode: bg supervision target value.")
    # MSST (Multi-Signal Sparse-Hard Trigger)
    parser.add_argument("--use-msst", action="store_true",
                         help="MSST: joint (d, smoothed-H) sparse-hard supervision on clu==2.")
    parser.add_argument("--lambda-msst", type=float, default=0.5,
                         help="MSST: loss weight.")
    parser.add_argument("--msst-d-fg-max", type=float, default=2.0,
                         help="MSST: d ≤ this triggers FG.")
    parser.add_argument("--msst-d-bg-min", type=float, default=10.0,
                         help="MSST: d > this triggers BG (in disjunction with low H).")
    parser.add_argument("--msst-h-kernel", type=int, default=5,
                         help="MSST: H smoothing kernel size (1=point, 5=diagnostic-sweet-spot).")
    parser.add_argument("--msst-h-fg-min", type=float, default=0.5,
                         help="MSST: normalized smoothed-H ≥ this triggers FG.")
    parser.add_argument("--msst-h-bg-max", type=float, default=0.3,
                         help="MSST: normalized smoothed-H < this triggers BG (in disjunction with high d).")
    parser.add_argument("--msst-target-fg", type=float, default=0.7,
                         help="MSST: supervision target value for FG-triggered pixels.")
    parser.add_argument("--msst-target-bg", type=float, default=0.0,
                         help="MSST: supervision target value for BG-triggered pixels.")
    # NICHE (Stratified Chromophore-Neutral Geometric Supervision)
    parser.add_argument("--use-niche", action="store_true",
                         help="NICHE: geometric supervision on chromophore-neutral clu==2 niche.")
    parser.add_argument("--lambda-niche", type=float, default=0.3,
                         help="NICHE: loss weight.")
    parser.add_argument("--niche-rho-max", type=float, default=0.2,
                         help="NICHE: |ρ| upper bound for chromophore-neutral.")
    parser.add_argument("--niche-d-fg-max", type=float, default=2.0,
                         help="NICHE: d ≤ this in neutral zone triggers fg target.")
    parser.add_argument("--niche-d-bg-min", type=float, default=12.0,
                         help="NICHE: d > this in neutral zone triggers bg target.")
    parser.add_argument("--niche-target-fg", type=float, default=0.7,
                         help="NICHE: fg supervision target.")
    parser.add_argument("--niche-target-bg", type=float, default=0.0,
                         help="NICHE: bg supervision target.")
    # IDG (Inter-Disk Gap)
    parser.add_argument("--use-idg", action="store_true",
                         help="IDG: touching-boundary bg supervision via geometric AND-gate + H valley.")
    parser.add_argument("--lambda-idg", type=float, default=0.3,
                         help="IDG: loss weight.")
    parser.add_argument("--idg-sigma", type=float, default=3.0,
                         help="IDG: exp decay scale on d_1, d_2 (px).")
    parser.add_argument("--idg-h-alpha", type=float, default=2.0,
                         help="IDG: power on (1-H_norm).")
    parser.add_argument("--idg-trigger-q", type=float, default=0.95,
                         help="IDG: top-quantile of T within clu==2 to trigger (default 0.95=top 5%%).")
    parser.add_argument("--idg-target", type=float, default=0.0,
                         help="IDG: supervision target for triggered pixels (default 0=bg).")
    # Boundary head
    parser.add_argument("--use-boundary", action="store_true",
                         help="Boundary head: geometric touching-boundary supervision + decode-time cellprob carving.")
    parser.add_argument("--lambda-boundary", type=float, default=0.5)
    parser.add_argument("--use-csbb", action="store_true",
                         help="CSBB-V3: boundary target = competing-sink ridge ∩ H-valley ∩ H>floor (EWB train-time analogue).")
    parser.add_argument("--csbb-rb", type=float, default=18.0)
    parser.add_argument("--csbb-h-floor", type=float, default=0.4)
    parser.add_argument("--use-gcsbb", action="store_true",
                         help="G-CSBB-v4 / CSSR: regime-gated competing-sink separation (G_patch*g_comp*g_fg).")
    parser.add_argument("--gcsbb-fg-tau", type=float, default=0.3)
    parser.add_argument("--gcsbb-crowd-r", type=float, default=18.0)
    parser.add_argument("--use-pace", action="store_true",
                         help="PACE: point-anchored chromophore-stable encoder reg (train-only).")
    parser.add_argument("--lambda-pace", type=float, default=0.05)
    parser.add_argument("--pace-tau", type=float, default=0.2)
    parser.add_argument("--pace-rn", type=float, default=48.0)
    parser.add_argument("--pace-eta", type=float, default=0.0)
    parser.add_argument("--pace-he", action="store_true")
    parser.add_argument("--pace-jit-sigma", type=float, default=0.08)
    parser.add_argument("--boundary-sigma", type=float, default=3.0)
    parser.add_argument("--boundary-decode-alpha", type=float, default=0.0,
                         help="At decode: cellprob -= alpha * sigmoid(boundary_logits). 0 disables.")
    parser.add_argument("--use-boundary-h", action="store_true",
                         help="V4_BND_H: modulate boundary target by |∇H̃|_norm.")
    parser.add_argument("--boundary-h-sigma", type=float, default=1.5,
                         help="V4_BND_H: Gaussian σ for H pre-gradient smoothing (px).")
    parser.add_argument("--boundary-h-floor", type=float, default=0.0,
                         help="V4_BND_H: pixels with |∇H̃|_norm < this → 0 modulation.")
    # CPR — Cross-Pixel Polarity Ranking (Module A, DSC paper)
    parser.add_argument("--use-cpr", action="store_true",
                          help="Enable CPR ranking loss on clu==2 pairs.")
    parser.add_argument("--lambda-cpr", type=float, default=0.0,
                          help="CPR loss weight.")
    parser.add_argument("--cpr-n-pairs", type=int, default=64,
                          help="Pairs per image per step.")
    parser.add_argument("--cpr-rho-diff-min", type=float, default=0.3,
                          help="Min ρ-gap to admit a pair.")
    parser.add_argument("--cpr-margin-init", type=float, default=0.05)
    parser.add_argument("--cpr-margin-final", type=float, default=0.25)
    parser.add_argument("--cpr-margin-warmup-epochs", type=int, default=40)
    # PEF — Polarity-Embedded Feature regularizer (Module B, DSC paper)
    parser.add_argument("--use-pef", action="store_true",
                          help="Enable PEF polarity-head regularizer on feat.")
    parser.add_argument("--lambda-pef", type=float, default=0.0,
                          help="PEF loss weight.")
    # CGFS — Chromophore-Guided Flow Supervision
    parser.add_argument("--use-dhs", action="store_true",
                         help="CGFS DHS: dense H̃ soft-label supervision on fg head over clu==2.")
    parser.add_argument("--lambda-dhs", type=float, default=1.0,
                         help="CGFS DHS: loss weight.")
    parser.add_argument("--dhs-sigma", type=float, default=1.5,
                         help="CGFS DHS: Gaussian smoothing σ for H̃.")
    parser.add_argument("--dhs-target-low", type=float, default=0.0,
                         help="CGFS DHS: H̃=0 maps to this target.")
    parser.add_argument("--dhs-target-high", type=float, default=1.0,
                         help="CGFS DHS: H̃=1 maps to this target.")
    parser.add_argument("--use-hgfa", action="store_true",
                         help="CGFS HGFA: ∇H̃ flow direction supervision on flow head over clu==2 ∩ d≤d_max.")
    parser.add_argument("--lambda-hgfa", type=float, default=1.0,
                         help="CGFS HGFA: loss weight.")
    parser.add_argument("--hgfa-sigma", type=float, default=1.5,
                         help="CGFS HGFA: Gaussian smoothing σ on H pre-gradient.")
    parser.add_argument("--hgfa-d-max", type=float, default=5.0,
                         help="CGFS HGFA: only supervise clu==2 pixels with d_to_clu1 ≤ this.")
    parser.add_argument("--hgfa-grad-floor", type=float, default=0.0,
                         help="CGFS HGFA: pixels with normalized |∇H̃| < this fraction → 0 weight.")
    parser.add_argument("--hgfa-weight-pow", type=float, default=1.0,
                         help="CGFS HGFA: power on |∇H̃| for supervision weight.")
    # E1 PCGS — Probabilistic Chromophore-Guided Supervision
    parser.add_argument("--use-pcgs", action="store_true",
                         help="E1 PCGS: replace L_stain_soft + L_eosin_neg with posterior-target BCE.")
    parser.add_argument("--pcgs-em-warmup", type=int, default=20,
                         help="E1 PCGS: epochs to keep hard-threshold supervision before EM kicks in.")
    parser.add_argument("--pcgs-ema", type=float, default=0.9,
                         help="E1 PCGS: EMA decay for EM parameter blending (0=no smoothing).")
    parser.add_argument("--pcgs-sigma-d-init", type=float, default=8.0,
                         help="E1 PCGS: initial proximity-Gaussian sigma in pixels.")
    parser.add_argument("--pcgs-prior-fg-init", type=float, default=0.35,
                         help="E1 PCGS: initial fg prior over voronoi-gap pixels.")
    parser.add_argument("--pcgs-lambda", type=float, default=1.0,
                         help="E1 PCGS: weight on the posterior BCE loss.")
    parser.add_argument("--pcgs-max-dist", type=float, default=30.0,
                         help="E1 PCGS: clamp on distance-to-nearest-point in pixels.")
    parser.add_argument("--pcgs-mode", choices=["anchor", "em", "v2"], default="anchor",
                         help="E1 PCGS: 'anchor' (default, stateless per-batch fit on "
                              "clu==1/clu==0 anchors) or 'em' (legacy EM, observed to drift).")
    parser.add_argument("--pcgs-bg-anchor-min-dist", type=float, default=40.0,
                         help="E1 PCGS anchor mode: min distance-to-point for bg anchor selection.")
    parser.add_argument("--pcgs-prior-fg", type=float, default=0.5,
                         help="E1 PCGS anchor mode: fg prior in posterior (not learned).")
    parser.add_argument("--pcgs-additive", action="store_true",
                         help="E1 PCGS: additive mode — keep L_stain_soft+L_eosin_neg ACTIVE (don't auto-zero), and add anchor PCGS as per-image probabilistic calibration on top of hard-threshold supervision.")
    parser.add_argument("--pcgs-target-temperature", type=float, default=1.0,
                         help="E1-T (route C.1): T>1 sharpens PCGS posterior target toward binary "
                              "for cross-domain robustness. T=1: raw posterior (E1.6 baseline). "
                              "T→∞: hard step at 0.5 (≈ C5 hard threshold). Recommended T=2.0.")
    # LR schedule
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                         help="LR schedule across epochs")
    parser.add_argument("--lr-min", type=float, default=1e-6,
                         help="Cosine schedule minimum LR")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="Per-epoch EMA decay for trainable params. 0.0 disables. Typical: 0.97.")
    parser.add_argument("--high-wd-value", type=float, default=3e-2,
                        help="E1-MSG v3: weight decay applied to parameters matching high_wd_keywords (currently shallow_proj). 100x base WD to keep them near zero-init.")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stain-aug", action="store_true",
                        help="Enable HED color jitter augmentation (Tellez et al. 2019) "
                             "to widen training stain distribution; helps cross-stain generalization.")
    parser.add_argument("--stain-aug-sigma", type=float, default=0.05,
                        help="Scale-perturbation std for HED jitter (default 0.05).")
    parser.add_argument("--stain-aug-bias", type=float, default=0.05,
                        help="Bias-perturbation std for HED jitter (default 0.05).")
    parser.add_argument("--point-subset-seed", type=int, default=0, help="Fixed original-image point-selection seed recorded by preprocessing")
    parser.add_argument("--point-ratio", type=float, default=1.0,
                        help="Fraction of points to keep (1.0=all). For annotation efficiency analysis.")
    parser.add_argument("--point-shift", type=int, default=0,
                        help="Max random shift per point in pixels (0=off). For robustness analysis.")
    parser.add_argument("--scale-jitter", action=argparse.BooleanOptionalAction, default=True,
                        help="Augment: rescale crop content to synthesize varied nucleus sizes (cross-domain scale robustness).")
    parser.add_argument("--scale-jitter-min", type=float, default=0.8)
    parser.add_argument("--scale-jitter-max", type=float, default=1.25)
    parser.add_argument("--use-sam-pseudo", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--sam-pseudo-path", default="")
    parser.add_argument("--stage2-pseudo-dir", default="",
                        help="Stage-2 self-train: dir with pseudo .npz files overriding instance_id field")
    parser.add_argument("--init-from", default="",
                        help="Stage-2 warm-start: path to stage-1 .pth ckpt (model_state will be loaded with strict=False)")
    parser.add_argument("--ckpt-split", choices=["val"], default="val",
                        help="Split used for best-checkpoint selection (default: val).")
    parser.add_argument("--input-mode", choices=["rgb", "h_repeat", "h_and_rgb", "he_rgb"], default="rgb",
                        help="rgb=3ch RGB; h_repeat=H replicated to 3ch; h_and_rgb=dual-input H-3ch + RGB-3ch; he_rgb=5ch [R,G,B,H,E].")
    parser.add_argument("--lambda-sam-fg", type=float, default=0.0)
    parser.add_argument("--lambda-sam-flow", type=float, default=0.0)
    parser.add_argument("--sam-erode-radius", type=int, default=1)
    parser.add_argument("--sam-area-min", type=int, default=15)
    parser.add_argument("--sam-area-max", type=int, default=5000)
    parser.add_argument("--flow-scale", type=float, default=70.0)
    parser.add_argument("--niter", type=int, default=30)
    parser.add_argument("--cellprob-mode", choices=["logits", "prob"], default="logits")
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--flow-threshold", type=float, default=0.0)
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--tile-batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--val-protocol", default="basic",
                         choices=["basic", "test-equiv"],
                         help="basic = current evaluate_model; test-equiv = also call decode_variants.py subprocess with S4_drop+TTA+dil1.")
    parser.add_argument("--also-eval-test", action=argparse.BooleanOptionalAction, default=False,
                        help="Each --eval-every, ALSO run full-decode evaluate_model on the test split (writes test_instance_metrics.csv).")
    parser.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--patience-evals", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--warmup-epochs", type=int, default=60)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-eval-images", type=int, default=0)
    parser.add_argument("--log-freq", type=int, default=25)
    parser.add_argument("--append-handoff", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()




def _run_test_equiv_decode(model, run_dir, epoch, dataset, ckpt_split, device, args):
    """Spawn decode_variants.py subprocess with S4_drop+TTA+dil1 on current model state.
    Returns AJI float or None on failure. Prints a stdout line on success."""
    import subprocess, json, shutil
    import torch as _torch
    import csv as _csv
    from pathlib import Path as _Path
    temp_dir = _Path(run_dir) / "_val_decode" / f"ep{epoch}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_ckpt = temp_dir / "_temp.pth"
    cfg = vars(args).copy()
    if "root" in cfg and not isinstance(cfg["root"], str):
        cfg["root"] = str(cfg["root"])
    save_obj = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "checkpoint": {"config": cfg, "epoch": epoch},
    }
    _torch.save(save_obj, temp_ckpt)
    cmd = [
        sys.executable, "-u", "-m", "chromaflow.decode_variants",
        "--root", str(args.root),
        "--dataset", dataset,
        "--split", ckpt_split,
        "--checkpoint", str(temp_ckpt),
        "--out-dir", str(temp_dir),
        "--variants", "S4_drop",
        "--tta", "hflip,vflip",
        "--mask-dilation", "1",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception as e:
        print(f"[val-test-equiv] subprocess error: {e}", flush=True)
        return None
    summary_csv = temp_dir / "summary.csv"
    if not summary_csv.exists():
        print(f"[val-test-equiv] decode failed; stderr (last 500c):\n{out.stderr[-500:]}", flush=True)
        return None
    try:
        with summary_csv.open() as f:
            rows = list(_csv.DictReader(f))
    except Exception as e:
        print(f"[val-test-equiv] csv parse error: {e}", flush=True)
        return None
    if not rows:
        return None
    # find S4_drop row for the requested split
    row = next((r for r in rows if r.get("variant", "") == "S4_drop"
                and r.get("split", "") == ckpt_split), rows[0])
    try:
        aji = float(row.get("AJI", "nan"))
        pq  = float(row.get("PQ", "nan"))
        dq  = float(row.get("DQ", "nan"))
        sq  = float(row.get("SQ", "nan"))
    except Exception:
        return None
    print(f"[val-test-equiv] epoch={epoch} split={ckpt_split} "
          f"AJI={aji:.4f} PQ={pq:.4f} DQ={dq:.4f} SQ={sq:.4f}", flush=True)
    # cleanup the temp ckpt to save disk
    try:
        temp_ckpt.unlink()
    except Exception:
        pass
    return aji


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    run_dir = make_run_dir(root, args)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = (run_dir / "stdout.log").open("a")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    set_seed(args.seed)
    snapshot_code(root, run_dir)
    inference_config = {
        **DEFAULT_INFERENCE_CONFIG,
        "cellprob_mode": args.cellprob_mode,
        "flow_scale": args.flow_scale,
        "niter": args.niter,
        "cellprob_threshold": args.cellprob_threshold,
        "flow_threshold": args.flow_threshold,
        "min_size": args.min_size,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "tile_batch_size": args.tile_batch_size,
        "input_mode": args.input_mode,
        "boundary_decode_alpha": float(getattr(args, "boundary_decode_alpha", 0.0)),
    }
    config = vars(args).copy()
    config["root"] = str(root)
    config["run_dir"] = str(run_dir)
    config["inference"] = inference_config
    config["checkpoint_selection"] = {
        "metric": "val_AJI",
        "source": "full_image_instance_inference",
        "postprocess": "S4_drop",
        "uses_train_loss": False,
        "uses_val_loss": False,
        "uses_val_fg_acc": False,
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    train_set = NucleiV3PatchDataset(
        root=root,
        dataset=args.dataset,
        subset="train",
        clu_subdir=args.clu_subdir,
        sags_snap_window=args.sags_snap_window,
        crop_size=args.crop_size,
        augment=args.augment,
        use_sam_pseudo=args.use_sam_pseudo,
        sam_pseudo_path=args.sam_pseudo_path or None,
        stage2_pseudo_dir=args.stage2_pseudo_dir or None,
        input_mode=args.input_mode,
        stain_aug=args.stain_aug,
        stain_aug_sigma=args.stain_aug_sigma,
        stain_aug_bias=args.stain_aug_bias,
        point_ratio=args.point_ratio,
        point_shift=args.point_shift,
        point_subset_seed=args.point_subset_seed,
        scale_jitter=args.scale_jitter,
        scale_jitter_min=args.scale_jitter_min,
        scale_jitter_max=args.scale_jitter_max,
    )
    shutil.copy2(root / "data/official_clean/ours_prepared" / args.dataset / "preparation.json", run_dir / "preparation.json")
    from chromaflow.protocol import split_path
    shutil.copy2(split_path(root, args.dataset), run_dir / "split_manifest.json")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} run_dir={run_dir}", flush=True)
    print(f"[setup] train_patches={len(train_set)}", flush=True)
    if args.encoder == "uni":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        print(f"[setup] using UNI encoder from {uni_path}", flush=True)
        model = NISFlowNetUNI(uni_path=uni_path, flow_param=args.flow_param).to(device)
    elif args.encoder == "sacaf":
        from chromaflow.model_sacaf import NISFlowNetSACAF
        print(f"[setup] using SACAF (Stain-Aware Cross-Attention Fusion) encoder", flush=True)
        model = NISFlowNetSACAF(pretrained=args.pretrained,
                                flow_param=args.flow_param).to(device)
    elif args.encoder == "r34dual":
        from chromaflow.model_r34dual import NISFlowNetR34Dual
        print(f"[setup] using R34Dual-simple (RGB+H dual, additive fusion)", flush=True)
        model = NISFlowNetR34Dual(pretrained=args.pretrained,
                                   flow_param=args.flow_param).to(device)
    elif args.encoder == "latef":
        from chromaflow.model_r34latef import NISFlowNetLateFusionR34
        print(f"[setup] using LateFusion-R34 (residual H at logit level only)", flush=True)
        model = NISFlowNetLateFusionR34(pretrained=args.pretrained,
                                         flow_param=args.flow_param).to(device)
    elif args.encoder == "dinov2":
        print(f"[setup] using DINOv2-ImageNet ViT-L/14 frozen encoder", flush=True)
        model = NISFlowNetDINOv2(flow_param=args.flow_param).to(device)
    elif args.encoder == "uni_hybrid_ln":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        print(f"[setup] using UNI hybrid + ChannelLayerNorm fusion (plan 1a) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridLN(uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained).to(device)
    elif args.encoder == "uni_hybrid":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        print(f"[setup] using UNI hybrid (R34-shallow + UNI-deep) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybrid(uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained).to(device)
    elif args.encoder == "uni_hybrid_iga":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        print(f"[setup] using UNI hybrid + Flow-Center IGA from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridIGA(uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained).to(device)
    elif args.encoder == "uni_hybrid_h":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        print(f"[setup] using H-UNI hybrid (H-R34 local + RGB-UNI semantic) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridH(uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained).to(device)
    elif args.encoder == "uni_hybrid_h_psc2f":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        print(f"[setup] using H-PS-C2F-Net (H-UNI hybrid + PS-C2F) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHPSC2F(uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained).to(device)
    elif args.encoder == "uni_hybrid_h_psic":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        print(f"[setup] using HUF-Net-V1 (H-UNI hybrid + PSIC contrast head) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHPSIC(uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained).to(device)
    elif args.encoder == "uni_hybrid_h_mta":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        print(f"[setup] using H2U-MTA (H-UNI hybrid + Morphology-guided Token Adapter, detach_h={args.mta_detach_h}, eta_init={args.mta_init_scale}) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHMTA(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            mta_hidden_ch=args.mta_hidden_ch, mta_detach_h=args.mta_detach_h,
            mta_init_scale=args.mta_init_scale,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_u2h_srg":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        print(f"[setup] using U2H-SRG (H-UNI hybrid + UNI-to-H Semantic Reliability Gate on s2, alpha={args.u2h_alpha}, detach_uni={args.u2h_detach_uni}) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHU2HSRG(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            u2h_alpha=args.u2h_alpha, u2h_hidden_ch=args.u2h_hidden_ch,
            u2h_detach_uni=args.u2h_detach_uni,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_ubim":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        print(f"[setup] using HU-BIM (H-UNI + Boundary Inference Module, lambda_bim={args.lambda_bim}, tau={args.bim_tau}) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHUBIM(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            bim_mid_ch=args.bim_mid_ch,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_ucaf":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        print(f"[setup] using HU-CAF (H-UNI + Cross-branch Aux Foreground, lambda_hucaf={args.lambda_hucaf}) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHUCAF(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_udac":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        print(f"[setup] using HU-DAC v2 (H-UNI + Disagreement-Aware Channel gating, mlp[2] small-Kaiming init, no-WD) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHUDAC(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            hudac_alpha=args.hudac_alpha, hudac_hidden=args.hudac_hidden,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_udar":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        print(f"[setup] using HU-DAR (H-UNI + Decoder Augmented Residual on s2; non-bypassable, no scalar gate, no-WD) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHUDAR(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_urcf":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        print(f"[setup] using RCF-Head (H-UNI + Ridge-Conditioned Flow Head, lambda_ridge={args.lambda_ridge}, sigma_b={args.ridge_sigma_b}) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHURCF(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            ridge_ch=args.ridge_ch,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_urcf_v2":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        print(f"[setup] using RCF-v2 (center-safe ridge residual on flow, lambda_ridge={args.lambda_ridge}, sigma_b={args.ridge_sigma_b}, r_core={args.ridge_r_core}, lambda_res={args.ridge_residual_scale}) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHURCFV2(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            ridge_ch=args.ridge_ch, residual_scale=args.ridge_residual_scale,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_lora":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        from chromaflow.uni_encoder_b1 import NISFlowNetUNIHybridHLoRA
        print(f"[setup] using B1 (H+UNI+LoRA r=16 alpha=16 on qkv) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHLoRA(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            lora_rank=16, lora_alpha=16, lora_dropout=0.1,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_lora_film":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        from chromaflow.uni_encoder_e1film import NISFlowNetUNIHybridHLoRAFiLM
        print(f"[setup] using E1-FiLM (B1 + PCGS-FiLM on 64-ch final feature) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHLoRAFiLM(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            lora_rank=16, lora_alpha=16, lora_dropout=0.1,
            film_hidden=64, bg_anchor_min_dist=40.0,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_lora_paa":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        from chromaflow.uni_encoder_e1paa import NISFlowNetUNIHybridHLoRAPaA
        print(f"[setup] using E1-PaA (B1 + PCGS posterior-as-attention) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHLoRAPaA(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            lora_rank=16, lora_alpha=16, lora_dropout=0.1,
            bg_anchor_min_dist=40.0,
        ).to(device)
    elif args.encoder == "mssp":
        from chromaflow.mssp_model import NISFlowNetMSSP
        print('[setup] using MSSP (RGB R34 + multi-scale H pyramid, no UNI)', flush=True)
        model = NISFlowNetMSSP(
            pretrained_r34=args.pretrained,
            flow_param=args.flow_param,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_lora_msg":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        from chromaflow.uni_encoder_e1msg import NISFlowNetUNIHybridHLoRAMSG
        print(f"[setup] using E1-MSG (B1 + multi-scale gating from PCGS entropy) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHLoRAMSG(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            lora_rank=16, lora_alpha=16, lora_dropout=0.1,
            bg_anchor_min_dist=40.0,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_lora_adapter":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        from chromaflow.uni_encoder_b2 import NISFlowNetUNIHybridHLoRAAdapter
        print(f"[setup] using B2 (B1 + cross-attn adapter @ /8 R34s2) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHLoRAAdapter(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            lora_rank=16, lora_alpha=16, lora_dropout=0.1,
            adapter_hidden_dim=256, adapter_num_heads=8, adapter_dropout=0.1, adapter_gamma_init=0.1,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_lora_hca":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        from chromaflow.uni_encoder_b25 import NISFlowNetUNIHybridHLoRAHCA
        print(f"[setup] using B2.5 (B1 + HCA hematoxylin-biased attn @ /8 R34s2) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHLoRAHCA(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            lora_rank=16, lora_alpha=16, lora_dropout=0.1,
            hca_hidden_dim=256, hca_num_heads=8, hca_dropout=0.1, hca_gamma_init=0.1, hca_h_bias_init=1.0,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_lora_sdca":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        from chromaflow.uni_encoder_d1 import NISFlowNetUNIHybridHLoRASDCA
        print(f"[setup] using D1 (SDCA stain-differential attn @ /8 R34s2) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHLoRASDCA(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            lora_rank=16, lora_alpha=16, lora_dropout=0.1,
            sdca_hidden_dim=256, sdca_num_heads=8, sdca_dropout=0.1,
            sdca_gamma_init=0.1, sdca_bias_init=2.0,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_lora_hca_r34rgb":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        from chromaflow.uni_encoder_d2 import NISFlowNetUNIHybridHLoRAHCA_R34RGB
        print(f"[setup] using D2 (HCA + R34-on-RGB) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHLoRAHCA_R34RGB(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            lora_rank=16, lora_alpha=16, lora_dropout=0.1,
            hca_hidden_dim=256, hca_num_heads=8, hca_dropout=0.1,
            hca_gamma_init=0.1, hca_h_bias_init=1.0,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_lora_adasca":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        from chromaflow.uni_encoder_d3 import NISFlowNetUNIHybridHLoRAAdaSCA
        print(f"[setup] using D3 (AdaSCA image-conditioned gate) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHLoRAAdaSCA(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            lora_rank=16, lora_alpha=16, lora_dropout=0.1,
            adasca_hidden_dim=256, adasca_num_heads=8, adasca_dropout=0.1,
            adasca_h_bias_init=1.0,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_lora_stainrecon":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        from chromaflow.uni_encoder_d7 import NISFlowNetUNIHybridHLoRAStainRecon
        print(f"[setup] using D7 (D4 backbone + StainRecon aux head) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHLoRAStainRecon(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            lora_rank=16, lora_alpha=16, lora_dropout=0.1,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_lora_evuc":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        from chromaflow.uni_encoder_d8 import NISFlowNetUNIHybridHLoRAEvUC
        print(f"[setup] using D8 (D4 backbone + evidential fg head) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHLoRAEvUC(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            lora_rank=16, lora_alpha=16, lora_dropout=0.1,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_lora_hca_hv":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        from chromaflow.uni_encoder_b5 import NISFlowNetUNIHybridHLoRAHCAHV
        print(f"[setup] using B5 (HCA + hv-map output + stain-weighted loss) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHLoRAHCAHV(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            lora_rank=16, lora_alpha=16, lora_dropout=0.1,
            hca_hidden_dim=256, hca_num_heads=8, hca_dropout=0.1, hca_gamma_init=0.1, hca_h_bias_init=1.0,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_lora_srma":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        from chromaflow.uni_encoder_srma import NISFlowNetUNIHybridHLoRASRMA
        print(f"[setup] using SRMA (Stain-Routed Mutual Attention, bidirectional FM<->CNN) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHLoRASRMA(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            lora_rank=16, lora_alpha=16, lora_dropout=0.1,
            srma_hidden_dim=256, srma_num_heads=8, srma_dropout=0.1,
            srma_gamma_init=0.1, srma_h_bias_init=1.0, srma_e_bias_init=1.0,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_lora_hca_v2":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        from chromaflow.uni_encoder_b25v2 import NISFlowNetUNIHybridHLoRAHCAv2
        print(f"[setup] using B2.5 v2 (H+E dual stain bias on attention) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHLoRAHCAv2(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            lora_rank=16, lora_alpha=16, lora_dropout=0.1,
            hca_hidden_dim=256, hca_num_heads=8, hca_dropout=0.1, hca_gamma_init=0.1,
            hca_h_bias_init=1.0, hca_e_bias_init=0.5,
        ).to(device)
    elif args.encoder == "uni_hybrid_h_lora_hca_mask":
        uni_path = args.uni_path if Path(args.uni_path).is_absolute() else str(Path(args.root) / args.uni_path)
        from chromaflow.uni_encoder_b3 import NISFlowNetUNIHybridHLoRAHCAMask
        print(f"[setup] using B3 (B2.5 + point-prompted mask decoder) from {uni_path}", flush=True)
        model = NISFlowNetUNIHybridHLoRAHCAMask(
            uni_path=uni_path, flow_param=args.flow_param, pretrained_r34=args.pretrained,
            lora_rank=16, lora_alpha=16, lora_dropout=0.1,
            hca_hidden_dim=256, hca_num_heads=8, hca_dropout=0.1, hca_gamma_init=0.1, hca_h_bias_init=1.0,
            mask_n_queries=96, mask_query_dim=128, mask_num_blocks=2, mask_num_heads=4, mask_dropout=0.1,
        ).to(device)
    elif args.encoder == "scde":
        from chromaflow.model_scde import NISFlowNetSCDE
        print(f"[setup] using SCDE (Shallow Chromophore-Disentangled Encoder)", flush=True)
        model = NISFlowNetSCDE(pretrained=args.pretrained, flow_param=args.flow_param).to(device)
    else:
        _in_ch = 5 if args.input_mode == "he_rgb" else 3
        _bb = args.encoder if args.encoder in ("resnet18", "resnet34", "resnet50", "resnet101") else "resnet34"
        model = NISFlowNet(pretrained=args.pretrained, flow_param=args.flow_param, in_channels=_in_ch,
                            with_boundary=getattr(args, "use_boundary", False), with_polarity=getattr(args, "use_pef", False), with_pace=getattr(args, "use_pace", False),
                            backbone=_bb).to(device)
        if hasattr(model, "decode_alpha"):
            model.decode_alpha = float(getattr(args, "boundary_decode_alpha", 0.0))
    # Stage-2 warm-start: load stage-1 weights (strict=False so newly-added heads like boundary stay random-init)
    if args.init_from:
        init_ckpt = torch.load(args.init_from, map_location=device)
        if isinstance(init_ckpt, dict):
            state = init_ckpt.get("model_state") or init_ckpt.get("state_dict") or init_ckpt
        else:
            state = init_ckpt
        cleaned = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"[setup] init-from {args.init_from}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if missing:
            print(f"[setup]   missing keys (kept random-init): {missing[:8]}{'...' if len(missing)>8 else ''}", flush=True)
    # Optimizer: split into no-WD group for module-named params. We mark
    # any param whose name contains one of these substrings as no-WD so
    # learnable gate / residual params don't get pulled back to zero by
    # weight decay (a root cause of MTA-v1 / HU-BIM v1 / HU-DAC v1 failures).
    no_wd_keywords = ["u2h_srg", "bim.", "hudac.", "udar_", "rcf.", "rcf2", "rcf2_res", "lora_", "lora_adapter.", "hca.", "mask_decoder.", "srma."]
    # E1-MSG v3: shallow_proj has *strong* WD to prevent it from drifting away
    # from zero-init; this preserves in-domain behaviour (where gate ≈ 1.0)
    # while still allowing shallow signal to emerge under high-entropy pixels.
    high_wd_keywords = ["shallow_proj"]
    high_wd_value = float(getattr(args, "high_wd_value", 3e-2))
    no_wd_params, base_params, high_wd_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(kw in n for kw in high_wd_keywords):
            high_wd_params.append(p)
        elif any(kw in n for kw in no_wd_keywords):
            no_wd_params.append(p)
        else:
            base_params.append(p)
    groups = [{"params": base_params, "lr": args.lr, "weight_decay": args.weight_decay}]
    if no_wd_params:
        groups.append({"params": no_wd_params, "lr": args.lr, "weight_decay": 0.0})
    if high_wd_params:
        groups.append({"params": high_wd_params, "lr": args.lr, "weight_decay": high_wd_value})
    if no_wd_params or high_wd_params:
        print(f"[setup] optimizer: base={len(base_params)} "
              f"+ no-WD={len(no_wd_params)} (WD=0.0) "
              f"+ high-WD={len(high_wd_params)} (WD={high_wd_value})", flush=True)
        optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    # EMA (optional): per-epoch shadow of trainable params for smoother inference.
    if args.ema_decay > 0:
        from chromaflow.ema import EMA
        ema = EMA(model, decay=args.ema_decay)
    else:
        ema = None

    # Optional cosine LR schedule (stabilizes late-training oscillation).
    # If args.lr_schedule == "cosine": lr decays from args.lr -> args.lr_min over args.epochs.
    lr_scheduler = None
    if getattr(args, "lr_schedule", "constant") == "cosine":
        eta_min = getattr(args, "lr_min", 1e-6)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=eta_min)
        print(f"[setup] cosine LR schedule: {args.lr} -> {eta_min} over {args.epochs} epochs", flush=True)

    # Checkpoint selection is configurable for legacy runs, but the canonical
    # protocol uses the disjoint validation split.
    ckpt_split = str(args.ckpt_split)
    if ckpt_split == "test":
        print(
            "[warning] --ckpt-split=test selects the checkpoint by test AJI. "
            "Use --ckpt-split=val for the canonical protocol.",
            flush=True,
        )
    best_metric = -float("inf")
    early_stop_reference = -float("inf")
    stale_eval_count = 0
    best_path = run_dir / f"best_{ckpt_split}_aji.pth"
    epoch_rows: list[dict[str, Any]] = []
    val_instance_rows: list[dict[str, Any]] = []

    # E1 PCGS: init state. Anchor mode (default) is stateless; EM mode (legacy)
    # carries (μ, σ, prior, σ_d) across epochs. Sealed inside `--use-pcgs`.
    pcgs_stats = None
    pcgs_anchor_mode = (getattr(args, "pcgs_mode", "anchor") == "anchor")
    if args.use_pcgs and not pcgs_anchor_mode:
        pcgs_stats = {
            "mu_H_fg":    0.65, "sigma_H_fg": 0.15,
            "mu_E_fg":    0.40, "sigma_E_fg": 0.20,   # on (1-E)
            "mu_H_bg":    0.30, "sigma_H_bg": 0.18,
            "mu_E_bg":    0.55, "sigma_E_bg": 0.20,
            "sigma_d":    float(args.pcgs_sigma_d_init),
            "prior_fg":   float(args.pcgs_prior_fg_init),
            "epoch_updated": -1,
            "history": [],
        }
        (run_dir / "pcgs_stats.json").write_text(json.dumps(pcgs_stats, indent=2))
        print(f"[pcgs em] init: warmup={args.pcgs_em_warmup} ema={args.pcgs_ema} "
              f"sigma_d={pcgs_stats['sigma_d']:.2f} prior_fg={pcgs_stats['prior_fg']:.2f}",
              flush=True)
    elif args.use_pcgs and pcgs_anchor_mode:
        print(
            f"[pcgs anchor] stateless mode: bg_anchor_min_dist={args.pcgs_bg_anchor_min_dist} "
            f"max_dist={args.pcgs_max_dist} prior_fg={args.pcgs_prior_fg} (no EM, no warmup)",
            flush=True,
        )

    # SPP: Stain Polarization Posterior state
    spp_state = None
    if getattr(args, "use_spp", False):
        from chromaflow.spp_loss import SPPState
        spp_state = SPPState(
            k=float(args.spp_k_init),
            b=float(args.spp_b_init),
            gamma=float(args.spp_gamma),
            ema_momentum=float(args.spp_ema_momentum),
        )
        print(f"[SPP] enabled: k={spp_state.k:.3f} b={spp_state.b:.3f} "
              f"gamma={spp_state.gamma:.2f} ema_m={spp_state.ema_momentum:.2f} "
              f"lambda={args.lambda_spp:.3f}", flush=True)

    for epoch in range(1, args.epochs + 1):
        print(f"[epoch {epoch}/{args.epochs}]", flush=True)
        if spp_state is not None and (epoch - 1) % int(args.spp_update_freq) == 0:
            spp_state.start_epoch()
        # EM buffer only relevant in legacy 'em' mode.
        pcgs_em_buffer = (
            {} if (args.use_pcgs and not pcgs_anchor_mode
                   and epoch >= int(args.pcgs_em_warmup))
            else None
        )
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, args, epoch=epoch,
            pcgs_stats=pcgs_stats, pcgs_em_buffer=pcgs_em_buffer,
            spp_state=spp_state,
        )
        # E1 PCGS M-step (em mode only): collapse accumulated sufficient statistics
        # to new (μ, σ, prior, σ_d) and EMA-blend into pcgs_stats. Snapshot history.
        if pcgs_em_buffer is not None and pcgs_em_buffer.get("n_fg", 0.0) > 1.0:
            from chromaflow.losses import pcgs_em_update_stats as _pcgs_M
            _pcgs_M(pcgs_stats, pcgs_em_buffer, ema=float(args.pcgs_ema))
            pcgs_stats["epoch_updated"] = epoch
            snapshot = {k: v for k, v in pcgs_stats.items() if k != "history"}
            snapshot["epoch"] = epoch
            pcgs_stats["history"].append(snapshot)
            (run_dir / "pcgs_stats.json").write_text(json.dumps(pcgs_stats, indent=2))
            print(
                f"[pcgs M-step e{epoch}] "
                f"mu_H_fg={pcgs_stats['mu_H_fg']:.3f} sigma_H_fg={pcgs_stats['sigma_H_fg']:.3f} "
                f"mu_H_bg={pcgs_stats['mu_H_bg']:.3f} sigma_d={pcgs_stats['sigma_d']:.2f} "
                f"prior_fg={pcgs_stats['prior_fg']:.3f}",
                flush=True,
            )

        # SPP: refit (k, b) at epoch end from anchor accumulator
        if spp_state is not None and (epoch - 1) % int(args.spp_update_freq) == 0:
            info = spp_state.update()
            print(f"[SPP] epoch={epoch} refit (k,b)=({spp_state.k:.3f},{spp_state.b:.3f}) "
                  f"AUC={info['auc']:.3f}  prev=({info['k_old']:.3f},{info['b_old']:.3f})",
                  flush=True)

        if ema is not None:
            ema.update(model)
        sel_aji = float("nan")
        if args.eval_every > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs):
            # Apply EMA weights for evaluation + ckpt save, then restore after the eval block.
            ema_backup = ema.apply_to(model) if ema is not None else None
            sel_summary, _ = evaluate_model(
                model,
                args.dataset,
                ckpt_split,
                root,
                inference_config,
                out_dir=None,
                save_predictions=False,
                limit=args.limit_eval_images,
                device=device,
            )
            sel_aji = float(sel_summary["AJI"])
            print(f"[{ckpt_split}] epoch={epoch} AJI={sel_aji:.4f} PQ={float(sel_summary['PQ']):.4f}", flush=True)
            if getattr(args, "val_protocol", "basic") == "test-equiv":
                try:
                    _run_test_equiv_decode(model, run_dir, epoch, args.dataset, ckpt_split, device, args)
                except Exception as _e:
                    print(f"[val-test-equiv] hook error: {_e}", flush=True)
            if args.also_eval_test and ckpt_split != "test":
                _test_summary, _ = evaluate_model(
                    model, args.dataset, "test", root, inference_config,
                    out_dir=None, save_predictions=False,
                    limit=args.limit_eval_images, device=device,
                )
                _test_aji = float(_test_summary["AJI"])
                print(f"[test ] epoch={epoch} AJI={_test_aji:.4f} PQ={float(_test_summary['PQ']):.4f}", flush=True)
                _test_row = {"epoch": epoch, **{f"test_{n}": float(_test_summary.get(n, float("nan"))) for n in METRIC_NAMES}}
                _test_rows_path = run_dir / "test_instance_metrics.csv"
                _existing = []
                if _test_rows_path.exists():
                    import csv as _csv
                    with _test_rows_path.open() as _f:
                        _existing = list(_csv.DictReader(_f))
                _existing.append({k: str(v) for k, v in _test_row.items()})
                with _test_rows_path.open("w", newline="") as _f:
                    import csv as _csv
                    _fn = ["epoch"] + [f"test_{n}" for n in METRIC_NAMES]
                    _w = _csv.DictWriter(_f, fieldnames=_fn)
                    _w.writeheader(); _w.writerows(_existing)
            # Keep stable `val_` column names regardless of ckpt_split choice (back-compat with CSV consumers).
            # Semantic: "val_" prefix here means "selection-split metric" (val OR test, per --ckpt-split).
            val_instance_rows.append(
                {
                    "epoch": epoch,
                    **{f"val_{name}": float(sel_summary.get(name, float("nan"))) for name in METRIC_NAMES},
                    "val_loss": float("nan"),
                    "val_fg_acc": float("nan"),
                    "train_loss": train_metrics["loss"],
                    "train_loss_fg": train_metrics["loss_fg"],
                    "train_loss_flow": train_metrics["loss_flow"],
                    "train_loss_point": train_metrics["loss_point"],
                    "train_loss_div": train_metrics["loss_div"],
                    "train_loss_sam_fg": train_metrics["loss_sam_fg"],
                    "train_loss_sam_flow": train_metrics["loss_sam_flow"],
                }
            )
            write_val_instance_metrics(run_dir / "val_instance_metrics.csv", val_instance_rows)
            if sel_aji > best_metric:
                best_metric = sel_aji
                save_checkpoint(best_path, model, optimizer, epoch, config, best_metric)
                print(f"[checkpoint] best_{ckpt_split}_aji={best_metric:.4f} path={best_path}", flush=True)
            if args.early_stop:
                if sel_aji > early_stop_reference + args.min_delta:
                    early_stop_reference = sel_aji
                    stale_eval_count = 0
                elif epoch >= args.warmup_epochs:
                    stale_eval_count += 1
                    print(
                        f"[early-stop] stale_eval_count={stale_eval_count}/{args.patience_evals} "
                        f"metric={ckpt_split}_AJI reference={early_stop_reference:.4f} current={sel_aji:.4f}",
                        flush=True,
                    )
            # Restore live (non-EMA) weights so the next training epoch continues from the optimizer state.
            if ema_backup is not None:
                ema.restore(model, ema_backup)
        save_checkpoint(run_dir / "last.pth", model, optimizer, epoch, config, best_metric)
        epoch_rows.append({"epoch": epoch, "val_AJI": sel_aji, **train_metrics})
        write_epoch_log(run_dir / "epoch_log.csv", epoch_rows)
        # cosine LR step (called once per epoch)
        if lr_scheduler is not None:
            lr_scheduler.step()
        if args.early_stop and epoch >= args.warmup_epochs and stale_eval_count >= args.patience_evals:
            print(
                f"[early-stop] stopping at epoch={epoch}; best_{ckpt_split}_aji={best_metric:.4f}; "
                f"no {ckpt_split}_AJI gain >= {args.min_delta} for {stale_eval_count} evals",
                flush=True,
            )
            break

    if not best_path.exists():
        shutil.copy2(run_dir / "last.pth", best_path)
    load_model_state(best_path, model, device)
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for split in ["val", "test"]:
        summary, rows = evaluate_model(
            model,
            args.dataset,
            split,
            root,
            inference_config,
            out_dir=run_dir,
            save_predictions=True,
            limit=args.limit_eval_images,
            device=device,
        )
        summaries[split] = summary
        all_rows.extend(rows)
    write_combined_per_image(run_dir, all_rows)
    (run_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    if args.append_handoff:
        append_handoff(root, run_dir, config, summaries)
    print(f"[done] best_{ckpt_split}_aji={best_metric:.4f} run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
