"""Stage-0 offline Poisson-flow probe v3 (ADDITIVE; does NOT touch training/model.py).

v2 modes (oracle, use GT masks): poisson_gt, poisson_center, poisson_gtflow, gtflow_direct.
v3 adds REALISTIC modes that use ONLY weak signals (GT points as a stand-in for the
already-accurate predicted centerness, plus the H stain channel) -- NO instance masks:
  pointstain_direct : flow = (unit vector toward nearest point) * H_weight ; decode directly.
  poisson_pointstain: rho = -div(that flow) ; Poisson-solve -> curl-free version.
These test whether points + stain carry enough SHAPE info to approach the shape-aware oracle.
"""
from __future__ import annotations
import numpy as np


def _rho_from_points(points_yx, H, W, sigma):
    out = np.zeros((H, W), np.float32)
    if points_yx is None or len(points_yx) == 0:
        return out
    yy = np.arange(H, dtype=np.float32)[:, None]
    xx = np.arange(W, dtype=np.float32)[None, :]
    s2 = 2.0 * sigma * sigma
    d2min = np.full((H, W), np.inf, np.float32)
    for p in points_yx:
        d2 = (yy - float(p[0])) ** 2 + (xx - float(p[1])) ** 2
        np.minimum(d2min, d2, out=d2min)
    return np.exp(-d2min / s2).astype(np.float32)


def _ideal_radial_flow(gt_inst):
    gt_inst = np.asarray(gt_inst)
    H, W = gt_inst.shape
    fx = np.zeros((H, W), np.float32); fy = np.zeros((H, W), np.float32)
    for iid in np.unique(gt_inst):
        if int(iid) == 0:
            continue
        ys, xs = np.where(gt_inst == iid)
        cy = ys.mean(); cx = xs.mean()
        dy = cy - ys.astype(np.float32); dx = cx - xs.astype(np.float32)
        n = np.sqrt(dx * dx + dy * dy) + 1e-6
        fx[ys, xs] = dx / n; fy[ys, xs] = dy / n
    return np.stack([fx, fy], axis=0)


def _voronoi_radial_flow_stain(gt_pts, H_chan, Hh, Ww, h_thresh=0.0, h_power=1.0):
    """Weak-signal shape-aware flow: assign each pixel to nearest GT point (voronoi),
    flow = unit vector toward that point, MAGNITUDE-WEIGHTED by normalized H (stain extent).
    Uses points + stain ONLY (no instance masks)."""
    pts = np.asarray(gt_pts, np.float32)
    fx = np.zeros((Hh, Ww), np.float32); fy = np.zeros((Hh, Ww), np.float32)
    if len(pts) == 0:
        return np.stack([fx, fy], axis=0)
    Y, X = np.indices((Hh, Ww)).astype(np.float32)
    d2min = np.full((Hh, Ww), np.inf, np.float32)
    idx = np.zeros((Hh, Ww), np.int32)
    for k in range(len(pts)):
        d2 = (Y - pts[k, 0]) ** 2 + (X - pts[k, 1]) ** 2
        m = d2 < d2min
        d2min = np.where(m, d2, d2min)
        idx = np.where(m, k, idx)
    cy = pts[idx, 0]; cx = pts[idx, 1]
    dy = cy - Y; dx = cx - X
    n = np.sqrt(dx * dx + dy * dy) + 1e-6
    Hn = (H_chan - H_chan.min()) / max(float(H_chan.max() - H_chan.min()), 1e-6)
    w = np.where(Hn > h_thresh, Hn ** h_power, 0.0).astype(np.float32)
    return np.stack([(dx / n) * w, (dy / n) * w], axis=0)


def _div_central(flow):
    f0, f1 = flow[0], flow[1]
    dx = np.zeros_like(f0); dy = np.zeros_like(f1)
    dx[:, 1:-1] = 0.5 * (f0[:, 2:] - f0[:, :-2]); dx[:, 0] = f0[:, 1] - f0[:, 0]; dx[:, -1] = f0[:, -1] - f0[:, -2]
    dy[1:-1, :] = 0.5 * (f1[2:, :] - f1[:-2, :]); dy[0, :] = f1[1, :] - f1[0, :]; dy[-1, :] = f1[-1, :] - f1[-2, :]
    return dx + dy


def _solve_poisson_neumann(rho, pad=0):
    from scipy.fft import dctn, idctn
    if pad > 0:
        rho = np.pad(rho, pad, mode="reflect")
    H, W = rho.shape
    r = (rho - rho.mean()).astype(np.float64)
    rhat = dctn(r, type=2, norm="ortho")
    i = np.arange(H, dtype=np.float64)[:, None]
    j = np.arange(W, dtype=np.float64)[None, :]
    lam = (2.0 * np.cos(np.pi * i / H) - 2.0) + (2.0 * np.cos(np.pi * j / W) - 2.0)
    lam[0, 0] = 1.0
    phihat = rhat / lam
    phihat[0, 0] = 0.0
    phi = idctn(phihat, type=2, norm="ortho").astype(np.float32)
    if pad > 0:
        phi = phi[pad:-pad, pad:-pad]
    return phi


def _neg_grad(phi, mode="central"):
    dx = np.zeros_like(phi); dy = np.zeros_like(phi)
    if mode == "forward":
        dx[:, :-1] = phi[:, 1:] - phi[:, :-1]; dx[:, -1] = phi[:, -1] - phi[:, -2]
        dy[:-1, :] = phi[1:, :] - phi[:-1, :]; dy[-1, :] = phi[-1, :] - phi[-2, :]
    else:
        dx[:, 1:-1] = 0.5 * (phi[:, 2:] - phi[:, :-2]); dx[:, 0] = phi[:, 1] - phi[:, 0]; dx[:, -1] = phi[:, -1] - phi[:, -2]
        dy[1:-1, :] = 0.5 * (phi[2:, :] - phi[:-2, :]); dy[0, :] = phi[1, :] - phi[0, :]; dy[-1, :] = phi[-1, :] - phi[-2, :]
    return np.stack([-dx, -dy], axis=0)


def _match_mag(flow, fl):
    o99 = float(np.percentile(np.sqrt(fl[0] ** 2 + fl[1] ** 2), 99)) + 1e-8
    n99 = float(np.percentile(np.sqrt(flow[0] ** 2 + flow[1] ** 2), 99)) + 1e-8
    return (flow * (o99 / n99)).astype(np.float32)


def poisson_override_flow(fl, ce, gt_inst, gt_pts, cfg):
    mode = cfg.get("flow_from", "model")
    if mode == "model":
        return fl
    ce = np.asarray(ce, dtype=np.float32)
    if ce.ndim == 3:
        ce = ce[0]
    H, W = ce.shape
    grad = cfg.get("poisson_grad", "central"); pad = int(cfg.get("poisson_pad", 0))
    h_thresh = float(cfg.get("h_thresh", 0.0)); h_power = float(cfg.get("h_power", 1.0))

    # --- direct flow modes (no Poisson) ---
    if mode == "gtflow_direct":
        return _match_mag(_ideal_radial_flow(gt_inst), fl)
    if mode == "pointstain_direct":
        flow = _voronoi_radial_flow_stain(gt_pts, np.asarray(cfg["stain_h"], np.float32), H, W, h_thresh, h_power)
        return _match_mag(flow, fl)

    # --- Poisson modes: build source rho, then solve ---
    src_field = None
    if mode == "poisson_gt":
        rho = _rho_from_points(gt_pts, H, W, float(cfg.get("poisson_sigma", 3.0)))
    elif mode == "poisson_center":
        rho = 1.0 / (1.0 + np.exp(-ce))
    elif mode == "poisson_gtflow":
        src_field = _ideal_radial_flow(gt_inst); rho = -_div_central(src_field)
    elif mode == "poisson_pointstain":
        src_field = _voronoi_radial_flow_stain(gt_pts, np.asarray(cfg["stain_h"], np.float32), H, W, h_thresh, h_power)
        rho = -_div_central(src_field)
    else:
        return fl
    if not np.isfinite(rho).all() or np.abs(rho).max() <= 0:
        return fl

    phi = _solve_poisson_neumann(rho, pad=pad)
    flow = _neg_grad(phi, mode=grad)
    if src_field is not None:
        if float((flow * src_field).sum()) < 0:
            flow = -flow
    else:
        if float((_div_central(flow) * rho).mean()) > 0:
            flow = -flow
    return _match_mag(flow, fl)
