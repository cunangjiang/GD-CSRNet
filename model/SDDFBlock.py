import torch
import torch.nn as nn
import torch.nn.functional as F


class DWRefine(nn.Module):
    """
    Lightweight local refinement:
        DWConv -> GELU -> PWConv
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    def forward(self, x):
        return self.block(x)

class SDDF(nn.Module):
    """
    Structure-Detail Decoupled Multi-scale Fusion (SDDF)

    Main idea:
        1) Use the same-scale enhanced skip feature as the PRIMARY source
           for recovering fine-grained local details at the current decoding stage.
        2) Use selected multi-scale target features to build a structural context prior,
           which provides cross-scale anatomical/semantic guidance.
        3) Decouple structural information and detail information, and then
           re-integrate them into the current decoder feature x in an anchor-aware manner.

    Inputs:
        x       : current decoder feature            [B, C, H, W]
        skip    : same-scale enhanced feature        [B, C, H, W]
        feat_list = [t1, t2, t3, t4] target encoder features

    Output:
        fused feature                                [B, C, H, W]
    """
    def __init__(
        self,
        channels: int,
        src_dims=(32, 48, 64, 96),
        use_scales=(0, 1, 2),
        hidden_ratio=0.5,
    ):
        super().__init__()
        self.channels = channels
        self.src_dims = src_dims
        self.use_scales = use_scales

        hidden_dim = max(8, int(channels * hidden_ratio))

        # -------------------------------------------------
        # 1) Align selected target multi-scale features to current C/H/W
        # -------------------------------------------------
        self.align_proj = nn.ModuleDict()
        self.align_refine = nn.ModuleDict()
        for i in use_scales:
            self.align_proj[str(i)] = nn.Conv2d(src_dims[i], channels, kernel_size=1)
            self.align_refine[str(i)] = DWRefine(channels)
            
        # -------------------------------------------------
        # 2) Spatial-aware multi-scale fusion
        # predict a spatial gate map for each aligned scale
        # -------------------------------------------------
        self.spatial_gate = nn.ModuleDict()
        for i in use_scales:
            self.spatial_gate[str(i)] = nn.Sequential(
                nn.Conv2d(channels * 3, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, 1, kernel_size=1)
            )

        # fuse spatially gated multi-scale features
        self.ms_fuse = nn.Sequential(
            nn.Conv2d(channels * len(use_scales), channels, kernel_size=1),
            nn.GELU(),
            DWRefine(channels)
        )

        # -------------------------------------------------
        # 2) 结构提取：偏低频/大范围
        # -------------------------------------------------
        self.struct_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

        # -------------------------------------------------
        # 3) 细节提取：偏高频/边缘
        # -------------------------------------------------
        self.detail_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

        # -------------------------------------------------
        # 4) anchor-aware 注入
        # -------------------------------------------------
        self.struct_gate = nn.Sequential(
            nn.Linear(channels * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels)
        )
        self.detail_gate = nn.Sequential(
            nn.Linear(channels * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.GELU(),
            DWRefine(channels)
        )

        # -------------------------------------------------
        # 5) Final local refinement
        # -------------------------------------------------
        self.out_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    @staticmethod
    def _gap(x):
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def _align_one(self, feat, idx, size_hw):
        """
        feat: [B, Ci, Hi, Wi]
        -> [B, C, H, W]
        """
        z = self.align_proj[str(idx)](feat)
        if z.shape[-2:] != size_hw:
            z = F.interpolate(z, size=size_hw, mode='bilinear', align_corners=False)
        z = self.align_refine[str(idx)](z)
        return z

    def _compute_ms_context(self, x, feat_list):
        """
        Build spatial-aware multi-scale target context.
        Returns:
            ms_ctx        : [B, C, H, W]
        """
        B, C, H, W = x.shape
        aligned_feats = []
        gated_feats = []

        for idx in self.use_scales:
            z = self._align_one(feat_list[idx], idx, (H, W))
            aligned_feats.append(z)

            gate_in = torch.cat([x, z, (z - x).abs()], dim=1)   # [B, 3C, H, W]
            gate = torch.sigmoid(self.spatial_gate[str(idx)](gate_in))  # [B,1,H,W]

            z_gated = gate * z
            gated_feats.append(z_gated)

        ms_ctx = self.ms_fuse(torch.cat(gated_feats, dim=1))   # [B,C,H,W]

        return ms_ctx

    def forward(self, x, skip, feat_list):
        ms_ctx = self._compute_ms_context(x, feat_list)

        # 结构来自 ms_ctx
        struct_feat = self.struct_proj(ms_ctx)

        # 细节来自 skip
        detail_feat = skip - F.avg_pool2d(skip, kernel_size=3, stride=1, padding=1)
        detail_feat = self.detail_proj(detail_feat)

        # x 作为 anchor 决定注入强度
        g_struct = torch.sigmoid(
            self.struct_gate(torch.cat([self._gap(x), self._gap(struct_feat)], dim=1))
        ).unsqueeze(-1).unsqueeze(-1)

        g_detail = torch.sigmoid(
            self.detail_gate(torch.cat([self._gap(x), self._gap(detail_feat)], dim=1))
        ).unsqueeze(-1).unsqueeze(-1)

        struct_inj = g_struct * struct_feat
        detail_inj = g_detail * detail_feat

        out = self.fuse(torch.cat([x, struct_inj, detail_inj], dim=1))

        out = x + out
        out = out + self.out_refine(out)
        
        return out
    
# =========================================================
# simple test
# =========================================================
def flops_conv_linear_only(model, x, skip, feat_list):
    """
    只统计 Conv2d / Linear 的 FLOPs。
    注意：
      1. FLOPs = 2 * MACs
      2. GELU、Sigmoid、AdaptiveAvgPool2d、插值上采样、逐元素加减乘除 未计入
      3. 因此这是“部分 FLOPs”，主要用于和你自己不同 RGSMF 版本做相对比较
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
        _ = model(x, skip, feat_list)

    for h in hooks:
        h.remove()

    flops = 2 * macs
    return flops


if __name__ == "__main__":
    import time

    if not torch.cuda.is_available():
        raise RuntimeError("RGSMF simple test requires CUDA for fair timing measurement.")

    device = torch.device("cuda")

    # -------------------------------------------------
    # Example setting: current stage channels = 64, spatial size = 56x56
    # feat_list corresponds to [t1, t2, t3, t4]
    # Here we mimic a decoder stage whose current feature is 56x56
    # -------------------------------------------------
    x = torch.randn(2, 64, 56, 56, device=device)
    skip = torch.randn(2, 64, 56, 56, device=device)

    feat_list = [
        torch.randn(2, 32, 224, 224, device=device),  # t1
        torch.randn(2, 48, 112, 112, device=device),  # t2
        torch.randn(2, 64, 56, 56, device=device),    # t3
        torch.randn(2, 96, 28, 28, device=device),    # t4
    ]

    model = SDDF(
        channels=64,
        src_dims=(32, 48, 64, 96),
        use_scales=(1, 2, 3),   # for example: use t2, t3, t4
        hidden_ratio=0.5
    ).to(device)

    model.eval()

    # 1) 参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total_params / 1e6:.3f} M")

    # 2) FLOPs
    flops = flops_conv_linear_only(model, x, skip, feat_list)
    print(f"FLOPs (Conv/Linear only): {flops / 1e9:.3f} G")
    print("Note: GELU / Sigmoid / AdaptiveAvgPool2d / Interpolation / element-wise ops are NOT counted.")

    # 3) 预热
    warmup_iterations = 10
    num_iterations = 50

    with torch.no_grad():
        for _ in range(warmup_iterations):
            _, _ = model(x, skip, feat_list)
    torch.cuda.synchronize()

    # 4) 推理时间
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            out, scale_weights = model(x, skip, feat_list)
    torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / num_iterations * 1000.0

    # 5) 输出
    with torch.no_grad():
        out, scale_weights = model(x, skip, feat_list, return_scale_weights=True)
    torch.cuda.synchronize()

    print("x shape:", x.shape)
    print("skip shape:", skip.shape)
    print("feat shapes:", [f.shape for f in feat_list])
    print("out shape:", out.shape)
    print("scale_weights shape:", scale_weights.shape)
    print("scale_weights:", scale_weights)
    print(f"Inference Time: {avg_time_ms:.3f} ms "
          f"(device={device.type}, batch={x.shape[0]})")
