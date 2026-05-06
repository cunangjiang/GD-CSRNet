# -*- coding: utf-8 -*-
# 运行：python -m gatedeltaunet.DGCFBlock
"""
DGCFBlock.py

Delta-Gated Cross-modal Fusion Block for multi-contrast MRI SR.

Design:
    Input:
        target feature    T : [B, C, H, W]
        reference feature R : [B, C, H, W]

    Core steps:
        1) Dual normalization
        2) Target-conditioned affine modulation on reference
        3) Cross-modal 2D gated delta update
           - query/state from target
           - key/value from aligned reference
           - 2D propagation by four directional scans
        4) Local refinement branch
        5) Residual fusion back to target

Notes:
    - This file rewrites the original single-input GatedDelta2D into
      a cross-modal fusion version.
    - "GatedDelta2DBlock" is renamed to "DGCF".
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# fla modules
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor
    
# =========================================================
# 2D RoPE utilities
# =========================================================
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1)
    return x_rot.flatten(-2)


def build_1d_rope(
    pos: torch.Tensor,
    dim: int,
    theta: float = 10000.0,
    dtype=None,
    device=None
):
    assert dim % 2 == 0
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    sinus = torch.einsum("t,d->td", pos.to(torch.float32), inv_freq)
    sin = sinus.sin()
    cos = sinus.cos()
    cos = torch.repeat_interleave(cos, repeats=2, dim=-1).to(
        dtype=dtype if dtype is not None else torch.float32
    )
    sin = torch.repeat_interleave(sin, repeats=2, dim=-1).to(
        dtype=dtype if dtype is not None else torch.float32
    )
    return cos, sin


def apply_2d_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    H: int,
    W: int,
    theta: float = 10000.0
):
    """
    q, k: [B, T, Nh, Dh], T = H * W, Dh % 4 == 0
    """
    B, T, nH, Dh = q.shape
    assert T == H * W
    assert Dh % 2 == 0 and (Dh // 2) % 2 == 0, "For 2D RoPE, head_dim must be divisible by 4."

    d_half = Dh // 2
    qx, qy = q[..., :d_half], q[..., d_half:]
    kx, ky = k[..., :d_half], k[..., d_half:]

    yy = torch.arange(H, device=q.device).unsqueeze(1).repeat(1, W).reshape(-1)
    xx = torch.arange(W, device=q.device).unsqueeze(0).repeat(H, 1).reshape(-1)

    cos_x, sin_x = build_1d_rope(xx, d_half, theta=theta, dtype=q.dtype, device=q.device)
    cos_y, sin_y = build_1d_rope(yy, d_half, theta=theta, dtype=q.dtype, device=q.device)

    cos_x = cos_x.view(1, T, 1, d_half)
    sin_x = sin_x.view(1, T, 1, d_half)
    cos_y = cos_y.view(1, T, 1, d_half)
    sin_y = sin_y.view(1, T, 1, d_half)

    qx = qx * cos_x + rotate_half(qx) * sin_x
    qy = qy * cos_y + rotate_half(qy) * sin_y
    kx = kx * cos_x + rotate_half(kx) * sin_x
    ky = ky * cos_y + rotate_half(ky) * sin_y

    return torch.cat([qx, qy], dim=-1), torch.cat([kx, ky], dim=-1)


# =========================================================
# directional scan utilities
# =========================================================
def make_direction_perms(H: int, W: int, device=None):
    """
    Four scan orders:
        p0: row-major
        p1: reverse row-major
        p2: column-major
        p3: reverse column-major
    """
    T = H * W
    p0 = torch.arange(T, device=device, dtype=torch.long)
    p1 = p0.flip(0)
    mat = p0.view(H, W)
    p2 = mat.transpose(0, 1).contiguous().view(-1)
    p3 = p2.flip(0)

    def invert(p):
        inv = torch.empty_like(p)
        inv[p] = torch.arange(p.shape[0], device=p.device, dtype=p.dtype)
        return inv

    return (p0, p1, p2, p3), (invert(p0), invert(p1), invert(p2), invert(p3))


# =========================================================
# Core cross-modal DGCF module
# =========================================================

class DeltaGatedCrossModal2D(nn.Module):
    """
    Cross-modal 2D gated delta fusion core.

    Input:
        tar_bhwc: [B, H, W, C]
        ref_bhwc: [B, H, W, C]

    Output:
        fused feature: [B, H, W, C]

    Pipeline:
        1) LN(T), LN(R)
        2) target-conditioned affine modulation on R
        3) Q from T, K/V from aligned R
        4) 2D gated delta rule with 4 directional scans
        5) local refinement branch
        6) project and return fused result (without outer residual here)
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expand_v: float = 2.0,
        num_v_heads: Optional[int] = None,
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        allow_neg_eigval: bool = False,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.expand_v = float(expand_v)
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.norm_eps = norm_eps
        self.allow_neg_eigval = allow_neg_eigval
        self.rope_theta = rope_theta

        self.head_k_dim = hidden_size // num_heads
        assert self.head_k_dim % 4 == 0, "For 2D RoPE, head_dim must be divisible by 4."

        self.head_v_dim = int(self.head_k_dim * self.expand_v)
        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim

        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError("num_v_heads must be divisible by num_heads.")
        if not math.isclose(self.head_k_dim * self.expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError("expand_v invalid for head_v_dim.")
        if not math.isclose(self.num_v_heads * self.head_k_dim * self.expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError("expand_v invalid for value_dim.")

        # -------------------------------------------------
        # Step 1: dual norm + target-conditioned affine modulation
        # -------------------------------------------------
        self.norm_t = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.norm_r = nn.LayerNorm(hidden_size, eps=norm_eps)

        # ===== [MOD-1] =====
        # 保留 gamma / beta 的线性预测，但 forward 中改成残差式调制：
        # ref_align = (1 + gamma) * ref_n + beta
        self.mod_gamma = nn.Linear(hidden_size, hidden_size, bias=True)
        self.mod_beta = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # ===== [SPATIAL-GUIDE-MOD-1] =====
        # Spatial cooperative guidance:
        # incoming guidance is now a feature map [B, C, H, W], not a global vector.
        # It is used to modulate current-layer alignment / beta gate / output gate.
        self.guide_norm = nn.LayerNorm(hidden_size, eps=norm_eps)

        # guidance -> spatial gamma / beta maps   [B,C,H,W]
        self.guide_to_gamma = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, bias=True)
        )
        self.guide_to_beta = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, bias=True)
        )

        # guidance -> head-wise beta gate bias   [B, num_v_heads, H, W]
        self.guide_to_beta_gate = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_size, self.num_v_heads, kernel_size=3, padding=1, bias=True)
        )

        if use_gate:
            # guidance -> value gate bias   [B, value_dim, H, W]
            self.guide_to_out_gate = nn.Sequential(
                nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, bias=True),
                nn.GELU(),
                nn.Conv2d(hidden_size, self.value_dim, kernel_size=3, padding=1, bias=True)
            )
        else:
            self.guide_to_out_gate = None

        # current layer outputs a new spatial guidance feature map
        self.guide_head = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, bias=True)
        )

        # zero init the last convs for stable warm start
        nn.init.zeros_(self.guide_to_gamma[-1].weight)
        nn.init.zeros_(self.guide_to_gamma[-1].bias)
        nn.init.zeros_(self.guide_to_beta[-1].weight)
        nn.init.zeros_(self.guide_to_beta[-1].bias)
        nn.init.zeros_(self.guide_to_beta_gate[-1].weight)
        nn.init.zeros_(self.guide_to_beta_gate[-1].bias)
        if self.guide_to_out_gate is not None:
            nn.init.zeros_(self.guide_to_out_gate[-1].weight)
            nn.init.zeros_(self.guide_to_out_gate[-1].bias)
            
        # ===== [OUT-RESIDUAL-MOD-1] =====
        # only apply residual scaling to out guidance
        # initialize with 0.5 to keep most of the useful effect of out guidance,
        # while reducing its over-strong injection tendency
        self.alpha_out = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # -------------------------------------------------
        # Step 2: cross projection
        # Q from target, K/V from aligned reference
        # -------------------------------------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # explicit difference-guided gate branch
        # ===== [MOD-2] =====
        # 差异分支改为绝对差描述：use |tar_n - ref_align|
        self.diff_proj = nn.Linear(hidden_size * 3, hidden_size, bias=False)
        # 这里虽然保留原变量名，但实际更接近 token-wise / head-wise reliability gate
        self.spatial_gate = nn.Linear(hidden_size, self.num_v_heads, bias=True)

        # dynamics projections
        # use target-reference mixed descriptor to generate a, b
        self.a_proj = nn.Linear(hidden_size * 2, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size * 2, self.num_v_heads, bias=False)

        # dynamics parameters
        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # -------------------------------------------------
        # optional short conv
        # -------------------------------------------------
        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
        else:
            self.q_conv1d = None
            self.k_conv1d = None
            self.v_conv1d = None

        # -------------------------------------------------
        # output gating
        # -------------------------------------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.g_proj = None
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
            
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # -------------------------------------------------
        # Step 4: local refinement branch
        # -------------------------------------------------
        self.local_dw = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)
        self.local_pw = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)

        # fuse global delta branch + local branch
        self.out_proj = nn.Linear(hidden_size + hidden_size, hidden_size, bias=False)
        
        nn.init.zeros_(self.mod_gamma.weight)
        nn.init.zeros_(self.mod_gamma.bias)
        nn.init.zeros_(self.mod_beta.weight)
        nn.init.zeros_(self.mod_beta.bias)

    @staticmethod
    def _conv_forward(mod: Optional[ShortConvolution], x: torch.Tensor):
        if mod is None:
            return F.silu(x)
        out = mod(x)
        return out[0] if isinstance(out, (tuple, list)) else out

    def forward(
        self, 
        tar_bhwc: torch.Tensor, 
        ref_bhwc: torch.Tensor,
        guide_feat: Optional[torch.Tensor] = None,
        return_guidance: bool = False
    ):
        """
        tar_bhwc: [B, H, W, C]
        ref_bhwc: [B, H, W, C]
        return:   [B, H, W, C]
        """
        B, H, W, C = tar_bhwc.shape
        T = H * W

        tar = tar_bhwc.view(B, T, C)
        ref = ref_bhwc.view(B, T, C)

        # =================================================
        # Step 1. LN-based dual normalization
        # =================================================
        tar_n = self.norm_t(tar)
        ref_n = self.norm_r(ref)

        # ===== [MOD-1] =====
        # 残差式调制，而不是直接 gamma * ref_n + beta
        # 更稳定，也更容易解释为“在 reference 上做条件微调”
        gamma = self.mod_gamma(tar_n)
        beta = self.mod_beta(tar_n)

        # ===== [SPATIAL-GUIDE-MOD-2] =====
        # incoming spatial guidance modulates current-layer alignment
        if guide_feat is not None:
            # guide_feat: [B, C, H, W]
            guide_feat_bhwc = guide_feat.permute(0, 2, 3, 1).contiguous()   # [B,H,W,C]
            guide_feat_bhwc = self.guide_norm(guide_feat_bhwc)
            guide_feat = guide_feat_bhwc.permute(0, 3, 1, 2).contiguous()   # [B,C,H,W]

            guide_gamma = self.guide_to_gamma(guide_feat)   # [B,C,H,W]
            guide_beta = self.guide_to_beta(guide_feat)     # [B,C,H,W]

            guide_gamma = guide_gamma.permute(0, 2, 3, 1).contiguous().view(B, T, C)
            guide_beta = guide_beta.permute(0, 2, 3, 1).contiguous().view(B, T, C)

            gamma = gamma + guide_gamma
            beta = beta + guide_beta

        ref_align = (1.0 + gamma) * ref_n + beta

        # explicit difference-aware descriptor
        diff_feat = torch.cat([tar_n, ref_align, torch.abs(tar_n - ref_align)], dim=-1)
        diff_feat = self.diff_proj(diff_feat)

        # =================================================
        # Step 2. Cross projection
        # =================================================
        q_lin = self._conv_forward(self.q_conv1d, self.q_proj(tar_n))
        k_lin = self._conv_forward(self.k_conv1d, self.k_proj(ref_align))
        v_lin = self._conv_forward(self.v_conv1d, self.v_proj(ref_align))

        q = q_lin.view(B, T, self.num_heads, self.head_k_dim)
        k = k_lin.view(B, T, self.num_heads, self.head_k_dim)
        v = v_lin.view(B, T, self.num_v_heads, self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            g_expand = self.num_v_heads // self.num_heads
            q = q.unsqueeze(3).expand(B, T, self.num_heads, g_expand, self.head_k_dim)
            q = q.reshape(B, T, self.num_v_heads, self.head_k_dim)

            k = k.unsqueeze(3).expand(B, T, self.num_heads, g_expand, self.head_k_dim)
            k = k.reshape(B, T, self.num_v_heads, self.head_k_dim)

        # 2D RoPE for spatial encoding
        q, k = apply_2d_rope(q, k, H, W, theta=self.rope_theta)

        # =================================================
        # Step 3. Cross-modal gated delta update
        # =================================================
        dyn_in = torch.cat([tar_n, ref_align], dim=-1)
        a = self.a_proj(dyn_in)
        b = self.b_proj(dyn_in)

        # gate: suppress unreliable reference injection
        # larger beta -> more update contribution
        # smaller beta -> more conservative propagation
        reliability = torch.sigmoid(self.spatial_gate(diff_feat))

        # ===== [SPATIAL-GUIDE-MOD-3] =====
        # incoming spatial guidance modulates current delta update strength
        if guide_feat is not None:
            guide_beta_gate = self.guide_to_beta_gate(guide_feat)   # [B, num_v_heads, H, W]
            guide_beta_gate = guide_beta_gate.permute(0, 2, 3, 1).contiguous().view(B, T, self.num_v_heads)
            beta_gate = torch.sigmoid(b + guide_beta_gate) * reliability
        else:
            beta_gate = torch.sigmoid(b) * reliability
        
        if self.allow_neg_eigval:
            beta_gate = beta_gate * 2.0

        g_dyn = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # four directional scans
        (p0, p1, p2, p3), (inv0, inv1, inv2, inv3) = make_direction_perms(H, W, device=tar.device)

        def gather_dir(tensor, p):
            return tensor.index_select(dim=1, index=p)

        q_dirs = [gather_dir(q, p) for p in (p0, p1, p2, p3)]
        k_dirs = [gather_dir(k, p) for p in (p0, p1, p2, p3)]
        v_dirs = [gather_dir(v, p) for p in (p0, p1, p2, p3)]
        g_dirs = [gather_dir(g_dyn, p) for p in (p0, p1, p2, p3)]
        beta_dirs = [gather_dir(beta_gate, p) for p in (p0, p1, p2, p3)]

        q_cat = torch.cat(q_dirs, dim=0).to(torch.bfloat16)
        k_cat = torch.cat(k_dirs, dim=0).to(torch.bfloat16)
        v_cat = torch.cat(v_dirs, dim=0).to(torch.bfloat16)
        g_cat = torch.cat(g_dirs, dim=0).to(torch.bfloat16)
        beta_cat = torch.cat(beta_dirs, dim=0).to(torch.bfloat16)
        y_cat, _ = chunk_gated_delta_rule(
            q=q_cat,
            k=k_cat,
            v=v_cat,
            g=g_cat,
            beta=beta_cat,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True
        )
        y_cat = y_cat.to(tar.dtype)
        y_dirs = torch.tensor_split(y_cat, 4, dim=0)

        y_sum = 0.0
        for y_d, invp in zip(y_dirs, (inv0, inv1, inv2, inv3)): 
            y_sum = y_sum + y_d.index_select(dim=1, index=invp)

        # output norm/gate
        if self.use_gate:
            g_out = self.g_proj(tar_n)   # [B, T, value_dim]

            # ===== [SPATIAL-GUIDE-MOD-4] =====
            # incoming spatial guidance modulates output gate
            if guide_feat is not None and self.guide_to_out_gate is not None:
                guide_out = self.guide_to_out_gate(guide_feat)   # [B, value_dim, H, W]
                guide_out = guide_out.permute(0, 2, 3, 1).contiguous().view(B, T, self.value_dim)

                # ===== [OUT-RESIDUAL-MOD-2] =====
                # only out guidance is residual-scaled
                g_out = g_out + self.alpha_out * guide_out

            g_out = g_out.view(B, T, self.num_v_heads, self.head_v_dim)
            y_delta = self.o_norm(y_sum, g_out)
        else:
            y_delta = self.o_norm(y_sum)

        y_delta = y_delta.view(B, T, self.value_dim)
        y_delta = self.o_proj(y_delta)
        # project delta branch back to hidden_size
        y_delta = y_delta.view(B, H, W, self.hidden_size)

        # =================================================
        # Step 4. local refinement branch
        # =================================================
        tar_chw = tar_bhwc.permute(0, 3, 1, 2).contiguous()
        refa_chw = ref_align.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        y_local = self.local_dw(tar_chw + refa_chw)
        y_local = F.gelu(y_local)
        y_local = self.local_pw(y_local)
        y_local = y_local.permute(0, 2, 3, 1).contiguous()

        # fuse delta branch + local branch
        y = torch.cat([y_delta, y_local], dim=-1)
        y = self.out_proj(y)   # [B, H, W, C]

        # ===== [SPATIAL-GUIDE-MOD-5] =====
        # output a new spatial guidance feature map for the next layer
        y_chw = y.permute(0, 3, 1, 2).contiguous()   # [B,C,H,W]
        guide_out = self.guide_head(y_chw)           # [B,C,H,W]

        if return_guidance:
            return y, guide_out
        return y

# =========================================================
# Wrapper block: format transform + residual
# =========================================================

class DGCF(nn.Module):
    """
    DGCF wrapper block.

    Input:
        tar_feat: [B, C, H, W]
        ref_feat: [B, C, H, W]

    Output:
        fused feature: [B, C, H, W]

    This class plays the role of the original GatedDelta2DBlock:
        - convert format BCHW -> BHWC
        - call cross-modal core
        - convert back
        - add residual to target branch
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        expand_v: float = 2.0,
        num_v_heads: Optional[int] = None,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        allow_neg_eigval: bool = False,
        rope_theta: float = 10000.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dgcf = DeltaGatedCrossModal2D(
            hidden_size=channels,
            num_heads=num_heads,
            expand_v=expand_v,
            num_v_heads=num_v_heads,
            use_gate=True,
            use_short_conv=True,
            conv_size=conv_size,
            conv_bias=conv_bias,
            norm_eps=norm_eps,
            allow_neg_eigval=allow_neg_eigval,
            rope_theta=rope_theta,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        tar_feat: torch.Tensor,
        ref_feat: torch.Tensor,
        guide_feat: Optional[torch.Tensor] = None,
        return_guidance: bool = False
    ):
        """
        tar_feat: [B, C, H, W]
        ref_feat: [B, C, H, W]
        guide_vec: [B, C] or None
        """
        tar_bhwc = tar_feat.permute(0, 2, 3, 1).contiguous()
        ref_bhwc = ref_feat.permute(0, 2, 3, 1).contiguous()

        # ===== [COOP-MOD-6] =====
        if return_guidance:
            y, new_guide = self.dgcf(
                tar_bhwc, ref_bhwc,
                guide_feat=guide_feat,
                return_guidance=True
            )
            y = self.drop_path(y).permute(0, 3, 1, 2).contiguous()
            out = tar_feat + y
            return out, new_guide

        y = self.dgcf(
            tar_bhwc, ref_bhwc,
            guide_feat=guide_feat,
            return_guidance=False
        )
        y = self.drop_path(y).permute(0, 3, 1, 2).contiguous()
        return tar_feat + y
    
# =========================================================
# simple test
# =========================================================
def flops_conv_linear_only(model, tar, ref):
    """
    只统计 Conv2d / Linear 的 FLOPs。
    注意：
      1. FLOPs = 2 * MACs
      2. fla 的 ShortConvolution、chunk_gated_delta_rule、Triton kernel 未计入
      3. 因此这是“部分 FLOPs”，主要用于和你自己不同 DGCF 版本做相对比较
    """
    macs = 0
    hooks = []

    def hook_fn(m, inp, out):
        nonlocal macs

        x0 = inp[0]
        y0 = out if not isinstance(out, (tuple, list)) else out[0]

        if isinstance(m, nn.Conv2d):
            # y0: [B, Cout, H, W]
            b, cout, ho, wo = y0.shape
            cin = m.in_channels
            kh, kw = m.kernel_size
            g = m.groups
            macs += b * cout * ho * wo * (cin // g) * kh * kw

        elif isinstance(m, nn.Linear):
            # 对最后一维做线性映射
            in_f = m.in_features
            out_f = m.out_features
            n = x0.numel() // in_f
            macs += n * in_f * out_f

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        _ = model(tar, ref)

    for h in hooks:
        h.remove()

    flops = 2 * macs
    return flops

if __name__ == "__main__":
    import time

    if not torch.cuda.is_available():
        raise RuntimeError("DGCFBlock simple test requires CUDA because fla/Triton kernels are used.")

    device = torch.device("cuda")

    tar = torch.randn(2, 64, 56, 56, device=device)
    ref = torch.randn(2, 64, 56, 56, device=device)

    model = DGCF(
        channels=64,
        num_heads=8,
        expand_v=2.0,
        num_v_heads=8,
        conv_size=4,
        rope_theta=10000.0,
        drop_path=0.0
    ).to(device)

    model.eval()

    # 1) 参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total_params / 1e6:.3f} M")

    # 2) FLOPs
    flops = flops_conv_linear_only(model, tar, ref)
    print(f"FLOPs (Conv/Linear only): {flops / 1e9:.3f} G")
    print("Note: ShortConvolution / chunk_gated_delta_rule / Triton custom kernels are NOT counted.")

    # 3) 预热
    warmup_iterations = 10
    num_iterations = 50

    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(tar, ref)
    torch.cuda.synchronize()

    # 4) 推理时间
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            out = model(tar, ref)
    torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / num_iterations * 1000

    # 5) 输出
    with torch.no_grad():
        out = model(tar, ref)
    torch.cuda.synchronize()

    print("tar shape:", tar.shape)
    print("ref shape:", ref.shape)
    print("out shape:", out.shape)
    print(f"Inference Time: {avg_time_ms:.3f} ms "
          f"(device={device.type}, batch={tar.shape[0]})")

# if __name__ == "__main__":
#     if not torch.cuda.is_available():
#         raise RuntimeError("CUDA required")

#     device = torch.device("cuda")

#     tar = torch.randn(1, 64, 56, 56, device=device)
#     ref = torch.randn(1, 64, 56, 56, device=device)

#     model = DGCF(
#         channels=64,
#         num_heads=8,
#         expand_v=2.0,
#         num_v_heads=8,
#         conv_size=4,
#         rope_theta=10000.0,
#         drop_path=0.0
#     ).to(device)

#     model.eval()

#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Params: {total_params / 1e6:.3f} M")
#     print("before forward")

#     with torch.no_grad():
#         out = model(tar, ref)

#     torch.cuda.synchronize()
#     print("after forward")
#     print("out shape:", out.shape)
