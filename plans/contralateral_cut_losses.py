"""
Combined generator objective for a contralateral (L-CC <-> R-CC) CUT model
trained on finding-free mammograms for downstream bilateral anomaly detection.

This module supplies ONLY the terms that are specific to this setup. It assumes
your existing CUT fork already provides, per direction:

    loss_gan        -- masked adversarial loss (your patched, mask-aware GANLoss)
    loss_nce_std    -- the standard PatchNCE (query=G(x), positives=x at same loc)

Those two keep their CUT defaults. Everything below is layered on top.

Design recap (why these terms, not others):
  * Single SHARED generator G applied both ways: fake_R = G(real_L), fake_L = G(real_R).
    G means "predict the expected normal contralateral", so every term is computed
    in BOTH directions and summed -> doubles supervision, prevents G specializing
    to one laterality.
  * Pairs are deformed (different breast shape), so NO raw L1 on unregistered breasts.
    Content is anchored by (a) the standard input PatchNCE you already have,
    (b) a contralateral-positive PatchNCE with local neighbourhood search,
    (c) an alignment-robust supervised term (Contextual loss on a feature extractor).
  * A high-frequency (gradient) term preserves microcalcifications -- but ONLY on a
    REGISTERED target, because it is a pixel-aligned comparison.
  * Identity term is OFF (it pushes G to copy its input -> erases the anomaly signal).
  * Cycle/involution term is OFF by default (pairs already make the map well-posed;
    its steganographic tendency manufactures contralateral false positives).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared masked reduction (your mask is already computed upstream; we just use it)
# ---------------------------------------------------------------------------
def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    mask = mask.expand_as(x)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def _mask_to(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Area-resample a [B,1,H,W] mask to a feature/grid resolution and re-binarize."""
    return (F.interpolate(mask.float(), size=(h, w), mode="area") > 0.5).float()


# ---------------------------------------------------------------------------
# 1. Contralateral-positive PatchNCE
#    Query = features of the GENERATED contralateral.
#    Positive = best match in a LOCAL WINDOW of the REAL contralateral's features
#               (not the identical coordinate -- the pair is deformed).
#    Negatives = other locations in the real contralateral's feature map.
# ---------------------------------------------------------------------------
class ContralateralPatchNCELoss(nn.Module):
    def __init__(self, nce_T: float = 0.07, radius: int = 2,
                 num_queries: int = 256, num_negatives: int = 255):
        super().__init__()
        self.nce_T = nce_T
        self.radius = radius                 # neighbourhood search radius (feature cells)
        self.num_queries = num_queries
        self.num_negatives = num_negatives
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, feat_q: torch.Tensor, feat_k: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        feat_q: [B, C, h, w] encoder features of the GENERATED breast  (e.g. G(L))
        feat_k: [B, C, h, w] encoder features of the REAL contralateral (e.g. real R)
        mask:   [B, 1, H, W] breast mask (resampled internally to h,w)
        """
        B, C, h, w = feat_q.shape
        dev, L, K = feat_q.device, h * w, (2 * self.radius + 1) ** 2

        q = F.normalize(feat_q, dim=1)
        k = F.normalize(feat_k, dim=1)
        k_unf = F.unfold(k, kernel_size=2 * self.radius + 1, padding=self.radius)
        k_unf = k_unf.view(B, C, K, L)        # local neighbourhood of every key location
        k_flat = k.view(B, C, L)
        q_flat = q.view(B, C, L)

        mflat = _mask_to(mask, h, w).view(B, L) if mask is not None else None

        losses = []
        for b in range(B):
            if mflat is not None:
                valid = mflat[b].nonzero(as_tuple=False).squeeze(1)
                if valid.numel() < 2:
                    continue
                P = min(self.num_queries, valid.numel())
                qid = valid[torch.randperm(valid.numel(), device=dev)[:P]]
                negpool = valid
            else:
                P = self.num_queries
                qid = torch.randint(0, L, (P,), device=dev)
                negpool = torch.arange(L, device=dev)

            qb = q_flat[b, :, qid].t()                          # [P, C]

            # positive = max similarity over the local window in the real contralateral
            neigh = k_unf[b, :, :, qid].permute(2, 1, 0)        # [P, K, C]
            pos = (neigh * qb.unsqueeze(1)).sum(-1).max(dim=1).values   # [P]

            # negatives = random real-contralateral locations
            N = min(self.num_negatives, negpool.numel())
            nid = negpool[torch.randint(0, negpool.numel(), (P, N), device=dev)]
            kneg = k_flat[b][:, nid].permute(1, 2, 0)           # [P, N, C]
            neg = (kneg * qb.unsqueeze(1)).sum(-1)              # [P, N]

            logits = torch.cat([pos.unsqueeze(1), neg], dim=1) / self.nce_T
            target = torch.zeros(P, dtype=torch.long, device=dev)
            losses.append(self.ce(logits, target).mean())

        if not losses:
            return feat_q.sum() * 0.0
        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# 2. Contextual loss (Mechrez et al. 2018) -- alignment-FREE supervised anchor.
#    Compares two feature sets without demanding pixel correspondence, so it
#    tolerates the L/R deformation. Feed it features from a feature extractor
#    (ideally lesion-sensitive / mammo-trained -- NOT generic ImageNet VGG).
# ---------------------------------------------------------------------------
class ContextualLoss(nn.Module):
    def __init__(self, band_width: float = 0.5, eps: float = 1e-5, max_points: int = 512):
        super().__init__()
        self.h = band_width
        self.eps = eps
        self.max_points = max_points

    def _subsample(self, t: torch.Tensor) -> torch.Tensor:
        if t.shape[0] > self.max_points:
            idx = torch.randperm(t.shape[0], device=t.device)[: self.max_points]
            t = t[idx]
        return t

    def forward(self, fx: torch.Tensor, fy: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """fx, fy: [B, C, H, W] feature maps of fake / real contralateral."""
        B, C, H, W = fx.shape
        total = fx.sum() * 0.0
        for b in range(B):
            x = fx[b].view(C, -1).t()
            y = fy[b].view(C, -1).t()
            if mask is not None:
                m = _mask_to(mask[b:b + 1], H, W).view(-1).bool()
                x, y = x[m], y[m]
                if x.numel() == 0 or y.numel() == 0:
                    continue
            x, y = self._subsample(x), self._subsample(y)

            mu = y.mean(0, keepdim=True)                 # CX feature centering
            x = F.normalize(x - mu, dim=1)
            y = F.normalize(y - mu, dim=1)

            d = 1.0 - x @ y.t()                          # cosine distance [Nx, Ny]
            d_tilde = d / (d.min(dim=1, keepdim=True).values + self.eps)
            w = torch.exp((1.0 - d_tilde) / self.h)
            A = w / (w.sum(dim=1, keepdim=True) + self.eps)
            cx = A.max(dim=1).values.mean()
            total = total + (-torch.log(cx + self.eps))
        return total / B


# ---------------------------------------------------------------------------
# 3. High-frequency (gradient) term -- microcalcification / edge preservation.
#    PIXEL-ALIGNED: only valid on a registered target (warped into fake's frame).
# ---------------------------------------------------------------------------
class GradientLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target_registered: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        t = target_registered
        lx = (pred[..., 1:] - pred[..., :-1] - (t[..., 1:] - t[..., :-1])).abs()
        ly = (pred[..., 1:, :] - pred[..., :-1, :] - (t[..., 1:, :] - t[..., :-1, :])).abs()
        if mask is not None:
            mx = mask[..., 1:] * mask[..., :-1]          # both neighbours in-mask
            my = mask[..., 1:, :] * mask[..., :-1, :]
            return masked_mean(lx, mx) + masked_mean(ly, my)
        return lx.mean() + ly.mean()


# ---------------------------------------------------------------------------
# Weights -- STARTING POINTS. The only principled tuning is to watch each logged
# term and balance gradient magnitudes; treat these as a first guess, not gospel.
# ---------------------------------------------------------------------------
@dataclass
class LossWeights:
    gan: float = 1.0          # CUT default (your existing masked GAN loss)
    nce: float = 1.0          # CUT default (your existing standard PatchNCE)
    nce_contra: float = 1.0   # same InfoNCE scale as nce
    sup: float = 1.0          # Contextual loss anchor; raise toward 2-3 if content drifts.
                              #   (If you instead use a REGISTERED pixel-L1 sup term,
                              #    its scale is ~10x smaller in effect -> use ~10.)
    hf: float = 1.0           # gradient term; raise toward ~5 if calcifications blur.
    cycle: float = 0.0        # involution G(G(x))~x. Keep 0; if used, <=0.5 + monitor leakage.
    idt: float = 0.0          # identity. Keep 0 -- it erases the anomaly signal.


@dataclass
class LossModules:
    nce_contra: ContralateralPatchNCELoss = field(default_factory=ContralateralPatchNCELoss)
    contextual: ContextualLoss = field(default_factory=ContextualLoss)
    gradient: GradientLoss = field(default_factory=GradientLoss)
    extractor: Optional[nn.Module] = None   # lesion-sensitive feature net for the sup term


# ---------------------------------------------------------------------------
# Per-direction assembly (called once for L->R and once for R->L)
# ---------------------------------------------------------------------------
def _direction(real_out: torch.Tensor, fake_out: torch.Tensor,
               feat_q: torch.Tensor, feat_k: torch.Tensor,
               mod: LossModules, w: LossWeights,
               mask_out: Optional[torch.Tensor] = None,
               target_registered: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """
    fake_out = G(real_in); real_out is the real contralateral target.
    feat_q   = encoder(fake_out); feat_k = encoder(real_out)   (for contralateral NCE)
    target_registered = real_out warped into fake_out's frame (None -> skip hf term)
    """
    terms: Dict[str, torch.Tensor] = {}
    terms["nce_contra"] = mod.nce_contra(feat_q, feat_k, mask=mask_out)

    assert mod.extractor is not None, "Provide a (mammo/lesion-sensitive) feature extractor"
    fx, fy = mod.extractor(fake_out), mod.extractor(real_out)
    terms["sup"] = mod.contextual(fx, fy, mask=mask_out)

    if target_registered is not None and w.hf > 0:
        terms["hf"] = mod.gradient(fake_out, target_registered, mask=mask_out)
    return terms


def generator_total_loss(
    loss_gan: torch.Tensor,        # from your masked GAN loss (sum of both directions, or per-dir)
    loss_nce_std: torch.Tensor,    # from your existing standard PatchNCE
    dir_LR: Dict[str, torch.Tensor],
    dir_RL: Dict[str, torch.Tensor],
    w: LossWeights,
    cycle_terms: Optional[Dict[str, torch.Tensor]] = None,  # {'L': G(G(L)) recon, ...}
):
    """Combine all terms; returns (total_loss, log_dict) so each term can be logged."""
    total = w.gan * loss_gan + w.nce * loss_nce_std
    log = {"gan": float(loss_gan.detach()), "nce_std": float(loss_nce_std.detach())}

    for tag, d in (("LR", dir_LR), ("RL", dir_RL)):
        total = total + w.nce_contra * d["nce_contra"] + w.sup * d["sup"]
        log[f"nce_contra_{tag}"] = float(d["nce_contra"].detach())
        log[f"sup_{tag}"] = float(d["sup"].detach())
        if "hf" in d:
            total = total + w.hf * d["hf"]
            log[f"hf_{tag}"] = float(d["hf"].detach())

    if w.cycle > 0 and cycle_terms is not None:
        cyc = sum(cycle_terms.values()) / max(len(cycle_terms), 1)
        total = total + w.cycle * cyc
        log["cycle"] = float(cyc.detach())

    log["total"] = float(total.detach())
    return total, log


# ---------------------------------------------------------------------------
# Sketch of one training step (pseudocode -- wire to your CUT fork):
# ---------------------------------------------------------------------------
#   fake_R, fake_L = G(real_L), G(real_R)                 # shared G, both directions
#   loss_gan       = gan_real_fake(...)                   # your masked GAN loss
#   loss_nce_std   = standard_patchnce(...)               # your existing CUT NCE
#
#   dir_LR = _direction(real_R, fake_R, enc(fake_R), enc(real_R), mod, w,
#                        mask_out=mask_R, target_registered=warp(real_R))
#   dir_RL = _direction(real_L, fake_L, enc(fake_L), enc(real_L), mod, w,
#                        mask_out=mask_L, target_registered=warp(real_L))
#
#   cyc = {"L": masked_mean((G(fake_R) - real_L).abs(), mask_L)} if w.cycle > 0 else None
#   loss_G, log = generator_total_loss(loss_gan, loss_nce_std, dir_LR, dir_RL, w, cyc)
#   loss_G.backward()
