'''
Gated Delta Cross-modal Structure-Refinement Network (GD-CSRNet)
GD-CSRNet: Gated Delta Cross-modal Structure Refinement for Multi-contrast MRI Super-Resolution
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# 同目录导入：既兼容包内导入，也兼容直接运行脚本
try:
    from .DGCFBlock import DropPath, DGCF
    from .SDDFBlock import SDDF
except ImportError:
    from DGCFBlock import DropPath, DGCF
    from SDDFBlock import SDDF


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x


def default_dgcf_heads(channels: int) -> int:
    """
    为 DGCF 自动选择一个合适的 num_heads。
    目标：尽量让每个 head 的维度为 16（同时满足被 4 整除，适配 2D RoPE）。

    对当前 dims=(32,48,64,96):
        32 -> 2
        48 -> 3
        64 -> 4
        96 -> 6
    """
    if channels % 16 == 0:
        heads = channels // 16
    elif channels % 12 == 0:
        heads = channels // 12
    elif channels % 8 == 0:
        heads = channels // 8
    elif channels % 4 == 0:
        heads = channels // 4
    else:
        heads = 1

    heads = max(1, heads)

    # 确保 channels // heads 能被 4 整除
    while channels % heads != 0 or (channels // heads) % 4 != 0:
        heads -= 1
        if heads == 1:
            break

    return heads


class LiteConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style lightweight block for MRI SR.
    Design:
      DWConv(7x7) -> LN -> PW expand -> GELU -> PW project -> layer scale -> residual
    """
    def __init__(self, dim, mlp_ratio=2.0, layer_scale_init_value=1e-6, drop_path=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) \
            if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = x * self.gamma[:, None, None]

        x = identity + self.drop_path(x)
        return x

class LiteConvNeXtStage(nn.Module):
    """
    Stage wrapper. Class name kept unchanged.
    """
    def __init__(self, dim, depth=2, mlp_ratio=2.0, drop_path=0.0):
        super().__init__()
        blocks = []
        for _ in range(depth):
            blocks.append(
                LiteConvNeXtBlock(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    layer_scale_init_value=1e-6,
                    drop_path=drop_path
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class ConvDownsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            LayerNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.proj(x)


class ModalityStem(nn.Module):
    """
    Separate shallow stem for target/ref to absorb modality-specific statistics.
    """
    def __init__(self, in_ch=1, out_ch=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.stem(x)


class LiteRefineBlock(nn.Module):
    """
    Lightweight decoder refinement block.
    Decoder is intentionally lighter than encoder.
    """
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)

class SharedEncoder(nn.Module):
    """
    Shared backbone encoder after modality-specific stems.
    Returns 4 scales.
    """
    def __init__(self, dims=(32, 48, 64, 96), depths=(2, 2, 2, 2), mlp_ratio=2.0):
        super().__init__()
        self.stage1 = LiteConvNeXtStage(dims[0], depth=depths[0], mlp_ratio=mlp_ratio)
        self.down1 = ConvDownsample(dims[0], dims[1])

        self.stage2 = LiteConvNeXtStage(dims[1], depth=depths[1], mlp_ratio=mlp_ratio)
        self.down2 = ConvDownsample(dims[1], dims[2])

        self.stage3 = LiteConvNeXtStage(dims[2], depth=depths[2], mlp_ratio=mlp_ratio)
        self.down3 = ConvDownsample(dims[2], dims[3])

        self.stage4 = LiteConvNeXtStage(dims[3], depth=depths[3], mlp_ratio=mlp_ratio)

    def forward(self, x):
        f1 = self.stage1(x)               # H, W
        f2 = self.stage2(self.down1(f1)) # H/2, W/2
        f3 = self.stage3(self.down2(f2)) # H/4, W/4
        f4 = self.stage4(self.down3(f3)) # H/8, W/8
        return [f1, f2, f3, f4]

class ProgressiveDecoder(nn.Module):
    """
    Non-symmetric lightweight decoder.

    Design:
      1) same-scale target-reference fusion by DGCF
      2) SDDF for Structure-Detail Decoupled Multi-scale Fusion
    """
    def __init__(self, dims=(32, 48, 64, 96), out_ch=1):
        super().__init__()
        c1, c2, c3, c4 = dims

        # project channels after upsample
        self.up3 = nn.Conv2d(c4, c3, kernel_size=1)
        self.up2 = nn.Conv2d(c3, c2, kernel_size=1)
        self.up1 = nn.Conv2d(c2, c1, kernel_size=1)
        
        # ===== [SPATIAL-GUIDE-MOD-6] =====
        # cross-scale spatial guidance projection
        self.guide_proj_32 = nn.Sequential(
            nn.Conv2d(c3, c2, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=True)
        )
        self.guide_proj_21 = nn.Sequential(
            nn.Conv2d(c2, c1, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=True)
        )

        self.refine3 = LiteRefineBlock(c3)
        self.refine2 = LiteRefineBlock(c2)
        self.refine1 = LiteRefineBlock(c1)

        # DGCF-based target-ref fusion per scale
        self.skip_fuse3 = DGCF(channels=c3, num_heads=default_dgcf_heads(c3))
        self.skip_fuse2 = DGCF(channels=c2, num_heads=default_dgcf_heads(c2))
        self.skip_fuse1 = DGCF(channels=c1, num_heads=default_dgcf_heads(c1))

        # SDDF blocks
        # stage3 (H/4): use t2, t3, t4
        self.ms_fuse3 = SDDF(
            channels=c3,
            src_dims=dims,
            use_scales=(1, 2, 3),
            hidden_ratio=0.5
        )

        # stage2 (H/2): use t1, t2, t3
        self.ms_fuse2 = SDDF(
            channels=c2,
            src_dims=dims,
            use_scales=(0, 1, 2),
            hidden_ratio=0.5
        )

        # stage1 (H): use t1, t2
        self.ms_fuse1 = SDDF(
            channels=c1,
            src_dims=dims,
            use_scales=(0, 1),
            hidden_ratio=0.5
        )

        self.out_head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c1, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, tar_feats, ref_feats, bottleneck):
        """
        tar_feats: [t1, t2, t3, t4]
        ref_feats: [r1, r2, r3, r4]
        bottleneck: fused deepest feature
        """
        t1, t2, t3, t4 = tar_feats
        r1, r2, r3, r4 = ref_feats
        

        # ===== [COOP-MOD-7] =====
        # coarse-to-fine cooperative DGCF
        # skip3 先产生 guidance，传给 skip2；skip2 再传给 skip1

        # stage 3: H/4
        x = F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up3(x)
        skip3, guide3 = self.skip_fuse3(t3, r3, guide_feat=None, return_guidance=True)
        x = self.ms_fuse3(x, skip3, [t1, t2, t3, t4])
        x = self.refine3(x)

        # stage 2: H/2
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up2(x)
        # ===== [SPATIAL-GUIDE-MOD-7] =====
        guide3_to_2 = F.interpolate(guide3, size=t2.shape[-2:], mode='bilinear', align_corners=False)
        guide3_to_2 = self.guide_proj_32(guide3_to_2)
        skip2, guide2 = self.skip_fuse2(t2, r2, guide_feat=guide3_to_2, return_guidance=True)
        x = self.ms_fuse2(x, skip2, [t1, t2, t3, t4])
        x = self.refine2(x)

        # stage 1: H
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up1(x)
        # ===== [SPATIAL-GUIDE-MOD-8] =====
        guide2_to_1 = F.interpolate(guide2, size=t1.shape[-2:], mode='bilinear', align_corners=False)
        guide2_to_1 = self.guide_proj_21(guide2_to_1)
        skip1, guide1 = self.skip_fuse1(t1, r1, guide_feat=guide2_to_1, return_guidance=True)
        x = self.ms_fuse1(x, skip1, [t1, t2, t3, t4])
        x = self.refine1(x)

        out = self.out_head(x)
        
        return out


    
class DualInputConvNeXtUNet(nn.Module):
    """
    A DCAMSR-style dual-input U-shape framework:
      - target branch stem
      - reference branch stem
      - shared 4-stage encoder backbone
      - deepest DGCF fusion
      - lightweight non-symmetric decoder with SDDF
    """
    def __init__(
        self,
        in_ch=1,
        out_ch=1,
        scale=4,
        dims=(48,72,96,144),
        depths=(2, 2, 2, 2),
        mlp_ratio=2.0
    ):
        super().__init__()
        self.scale = scale

        # shallow modality-specific stems
        self.tar_stem = ModalityStem(in_ch=in_ch, out_ch=dims[0])
        self.ref_stem = ModalityStem(in_ch=in_ch, out_ch=dims[0])

        # shared encoder backbone
        self.shared_encoder = SharedEncoder(
            dims=dims,
            depths=depths,
            mlp_ratio=mlp_ratio
        )

        # deepest DGCF fusion
        self.deep_fuse = DGCF(
            channels=dims[-1],
            num_heads=default_dgcf_heads(dims[-1])
        )

        # decoder
        self.decoder = ProgressiveDecoder(dims=dims, out_ch=out_ch)

    def encode_target(self, tar_up):
        x = self.tar_stem(tar_up)
        feats = self.shared_encoder(x)
        return feats

    def encode_reference(self, ref):
        x = self.ref_stem(ref)
        feats = self.shared_encoder(x)
        return feats

    def forward(self, tar_lr, ref):
        # upsample target LR to HR size first, similar to DCAMSR
        tar_up = F.interpolate(tar_lr, scale_factor=self.scale, mode='bilinear', align_corners=False)

        tar_feats = self.encode_target(tar_up)  # [t1, t2, t3, t4]
        ref_feats = self.encode_reference(ref)  # [r1, r2, r3, r4]

        # deepest DGCF fusion
        bottleneck = self.deep_fuse(tar_feats[-1], ref_feats[-1])

        out = self.decoder(tar_feats, ref_feats, bottleneck)

        # global residual
        out = out + tar_up
        return out

# ========== quick test ==========
def flops_conv_linear_only(model, tar, ref):
    """
    只统计 Conv2d / Linear 的 FLOPs。
    注意：
      1. FLOPs = 2 * MACs
      2. fla 的 ShortConvolution、chunk_gated_delta_rule、Triton kernel 未计入
      3. SDDF 中的 GELU / Sigmoid / AdaptiveAvgPool2d / Interpolation / element-wise ops 未计入
      4. 因此这是“部分 FLOPs”，主要用于和你自己不同版本做相对比较
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
        raise RuntimeError(
            "gated_delta_sr simple test requires CUDA because DGCF uses fla/Triton kernels."
        )

    device = torch.device("cuda")

    tar = torch.randn(1, 3, 56, 56, device=device)
    ref = torch.randn(1, 3, 224, 224, device=device)

    model = DualInputConvNeXtUNet(
        in_ch=3,
        out_ch=3,
        scale=4,
        dims=(48,72,96,144),
        depths=(2, 2, 2, 2),
        mlp_ratio=2.0
    ).to(device)

    model.eval()

    # 1) 参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total_params / 1e6:.3f} M")

    # 2) FLOPs
    flops = flops_conv_linear_only(model, tar, ref)
    print(f"FLOPs (Conv/Linear only): {flops / 1e9:.3f} G")
    print("Note: DGCF internal ShortConvolution / chunk_gated_delta_rule / Triton custom kernels are NOT counted.")
    print("Note: SDDF internal GELU / Sigmoid / AdaptiveAvgPool2d / Interpolation / element-wise ops are NOT counted.")

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

    avg_time_ms = (end_time - start_time) / num_iterations * 1000.0

    # 5) 输出 shape
    with torch.no_grad():
        out = model(tar, ref)
    torch.cuda.synchronize()

    print("Device:", device)
    print("tar shape:", tar.shape)
    print("ref shape:", ref.shape)
    print("Output shape:", out.shape)
    print(f"Inference Time: {avg_time_ms:.3f} ms (device={device.type}, batch={tar.shape[0]})")
