# IGMSA is replaced by IFA for feature fusion.
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from .SS2D_arch import SS2D
from .IFA_arch import IFA


# Truncated-normal initializers, used to keep early-training activations in range.
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Fill ``tensor`` in place from a normal distribution truncated to [a, b]."""

    # Normal CDF.
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    # Warn when the mean sits more than 2 std outside [a, b].
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():  # in-place, no autograd
        # CDF values at the truncation bounds.
        l = norm_cdf((a - mean) / std)  # noqa: E741
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()  # inverse error function
        tensor.mul_(std * math.sqrt(2.))  # scale
        tensor.add_(mean)  # shift
        tensor.clamp_(min=a, max=b)  # truncate

        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Public wrapper around :func:`_no_grad_trunc_normal_`."""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    """Variance-scaling init: derive a variance from fan-in/fan-out, then draw from it."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    # Denominator selected by mode.
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    # Draw from the requested distribution.
    if distribution == "truncated_normal":
        # 0.87962566103423978 maps a truncated normal's std back to the underlying
        # normal's, so the two branches end up with matching variance.
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    """LeCun normal init: fan-in variance scaling with a truncated normal."""
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    """LayerNorm applied before ``fn``, the usual pre-norm Transformer arrangement."""

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    """GELU activation as a module, so it can sit inside ``nn.Sequential``."""

    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    """kxk convolution with padding that preserves the spatial size."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),  # keep H, W unchanged
        bias=bias,
        stride=stride
    )


def shift_back(inputs, step=2):
    """Shift each channel along the column axis and crop, keeping ``row`` columns."""
    [bs, nC, row, col] = inputs.shape

    # Fixed 256-column output, so the ratio is 256 // row.
    down_sample = 256 // row

    # The step is scaled by the squared ratio.
    step = float(step) / float(down_sample * down_sample)

    # Output width equals the input row count.
    out_col = row

    # Shift every channel by its own offset.
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]

    return inputs[:, :, :, :out_col]


class AdaptiveIlluminationPriorGenerator(nn.Module):
    """Module A: build a low-frequency illumination prior with adaptive scale selection."""

    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3, kernel_sizes=(3, 7, 15)):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        hidden_dim = max(8, n_fea_middle // 4)

        self.selector = nn.Sequential(
            nn.Conv2d(n_fea_in, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, len(kernel_sizes), kernel_size=1, bias=True))
        self.prior_refine = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True))
        self.feature_proj = nn.Conv2d(1, n_fea_middle, kernel_size=1, bias=True)
        self.map_proj = nn.Conv2d(1, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img, mean_c):
        guide = torch.cat([img, mean_c], dim=1)
        selector = torch.softmax(self.selector(guide), dim=1)

        priors = []
        for kernel_size in self.kernel_sizes:
            pad = kernel_size // 2
            pooled = F.avg_pool2d(
                F.pad(mean_c, (pad, pad, pad, pad), mode='replicate'),
                kernel_size=kernel_size,
                stride=1)
            priors.append(pooled)

        multi_scale_prior = torch.stack(priors, dim=1)
        adaptive_prior = (selector.unsqueeze(2) * multi_scale_prior).sum(dim=1)
        adaptive_prior = adaptive_prior + self.prior_refine(adaptive_prior)

        prior_fea = self.feature_proj(adaptive_prior)
        prior_map = self.map_proj(adaptive_prior)
        return prior_fea, prior_map


class AdaptiveCoordinateAttention(nn.Module):
    """A lightweight directional gate for boundary-sensitive feature refinement."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden_dim = max(8, channels // reduction)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv_h = nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)
        y = self.shared(torch.cat([x_h, x_w], dim=2))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)

        gate_h = torch.sigmoid(self.conv_h(y_h))
        gate_w = torch.sigmoid(self.conv_w(y_w))
        return x * gate_h * gate_w


class MultiScaleBoundaryAwareRefiner(nn.Module):
    """Module B: refine illumination features with multi-scale context and soft boundary gating."""

    def __init__(self, channels):
        super().__init__()
        self.guide_proj = nn.Conv2d(channels + 1, channels, kernel_size=1, bias=True)
        self.branch_3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.GELU())
        self.branch_5 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.GELU())
        self.branch_7 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.GELU())
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=True),
            nn.GELU())
        self.boundary_gate = AdaptiveCoordinateAttention(channels)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, mean_c):
        guide = self.guide_proj(torch.cat([x, mean_c], dim=1))
        multi_scale = torch.cat([
            self.branch_3(guide),
            self.branch_5(guide),
            self.branch_7(guide)
        ], dim=1)
        multi_scale = self.fuse(multi_scale)
        multi_scale = self.boundary_gate(multi_scale)
        return x + self.res_scale * self.out_proj(multi_scale)


class Illumination_Estimator(nn.Module):
    """Explicit Illumination Encoder (EIE): illumination features and map from one image."""

    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3,
                 illu_use_a=False, illu_use_b=False):
        super(Illumination_Estimator, self).__init__()
        self.illu_use_a = illu_use_a
        self.illu_use_b = illu_use_b

        # 1x1 projection into the middle feature space.
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        # Depthwise 5x5 for local spatial context.
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        # 1x1 projection to the illumination map.
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

        if self.illu_use_a:
            self.prior_generator = AdaptiveIlluminationPriorGenerator(
                n_fea_middle=n_fea_middle, n_fea_in=n_fea_in, n_fea_out=n_fea_out)
            self.prior_fea_scale = nn.Parameter(torch.tensor(0.1))
            self.prior_map_scale = nn.Parameter(torch.tensor(0.1))

        if self.illu_use_b:
            self.feature_refiner = MultiScaleBoundaryAwareRefiner(n_fea_middle)

    def forward(self, img):
        # Lp: the channel-mean intensity prior.
        mean_c = img.mean(dim=1).unsqueeze(1)  # (b, 1, h, w)

        # Concatenate the image with its channel mean.
        input = torch.cat([img, mean_c], dim=1)

        # Project, then optionally add the adaptive prior.
        x_1 = self.conv1(input)
        prior_map = None

        if self.illu_use_a:
            prior_fea, prior_map = self.prior_generator(img, mean_c)
            x_1 = x_1 + self.prior_fea_scale * prior_fea

        illu_fea = self.depth_conv(x_1)

        if self.illu_use_b:
            illu_fea = self.feature_refiner(illu_fea, mean_c)

        # Illumination map, optionally refined by the adaptive prior map.
        illu_map = self.conv2(illu_fea)
        if prior_map is not None:
            illu_map = illu_map + self.prior_map_scale * prior_map

        return illu_fea, illu_map


class GatedFeedForward(nn.Module):
    """Gated feed-forward: 1x1 expand, depthwise 3x3, gate, 1x1 project back."""

    def __init__(self, dim, mult=2.66):
        super().__init__()
        hidden_features = int(dim * mult)

        # Expand to 2x hidden so the output can be split into a gate pair.
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=False)

        # Depthwise 3x3 on the expanded features.
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=False)

        # Project back to dim.
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

    def forward(self, x):
        # IGAB hands this module [B, H, W, C], so permute before Conv2d.
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # gate pair
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        # Back to [B, H, W, C] for the residual add in IGAB.
        return x.permute(0, 2, 3, 1)


class SpatialEnhancedGatedFeedForward(nn.Module):
    """SEM-inspired FFN that injects low-frequency spatial guidance from illumination features."""

    def __init__(self, dim, mult=2.66, use_feature_path=True):
        super().__init__()
        # Fixed F-path gate; not part of state_dict and not learnable.
        self.feature_gate = 1.0 if use_feature_path else 0.0
        hidden_features = int(dim * mult)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1,
            padding=1, groups=hidden_features * 2, bias=False)

        self.spatial_proj = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=False)
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1,
                      groups=hidden_features, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=1, bias=False))

        self.fusion = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=1, bias=False)
        self.dwconv_after_fusion = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=3, stride=1,
            padding=1, groups=hidden_features, bias=False)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, spatial):
        x = x.permute(0, 3, 1, 2).contiguous()
        spatial = F.interpolate(spatial, size=x.shape[-2:], mode='bilinear', align_corners=False)

        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)

        spatial = self.spatial_proj(spatial)
        spatial = F.avg_pool2d(spatial, kernel_size=2, stride=2, ceil_mode=True)
        spatial = self.spatial_refine(spatial)
        spatial = F.interpolate(spatial, size=x1.shape[-2:], mode='bilinear', align_corners=False)

        # F-path gate for IGFFN: remove only estimator-produced guidance immediately
        # before fusion; the IGFFN main-feature branch and all module parameters remain.
        spatial = self.feature_gate * spatial

        x1 = self.fusion(torch.cat([x1, spatial], dim=1))
        x1 = self.dwconv_after_fusion(x1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return self.res_scale * x.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    """Interleaved Group Attention Block: IFA -> SS2D -> feed-forward, repeated."""

    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2, d_state=16, ffn_use_sem=False,
                 use_feature_path=True):

        super().__init__()
        self.ffn_use_sem = ffn_use_sem
        self.norm_ss2d = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            ff_module = SpatialEnhancedGatedFeedForward(
                dim=dim, mult=2.66, use_feature_path=use_feature_path) if ffn_use_sem else \
                GatedFeedForward(dim=dim, mult=2.66)
            self.blocks.append(nn.ModuleList([
                IFA(dim_2=dim, dim=dim, num_heads=heads, ffn_expansion_factor=2.66, bias=True,
                    LayerNorm_type='WithBias', use_feature_path=use_feature_path),
                SS2D(d_model=dim, dropout=0, d_state=d_state),  # upstream defaults
                PreNorm(dim, ff_module)  # LN + FFN
            ]))

    def forward(self, x, illu_fea):
        for (trans, ss2d, ff) in self.blocks:
            y = trans(x, illu_fea).permute(0, 2, 3, 1)
            # SS2D with a residual connection.
            x = ss2d(self.norm_ss2d(y)) + x.permute(0, 2, 3, 1)
            # Feed-forward with a residual connection.
            if self.ffn_use_sem:
                x = ff(x, illu_fea) + x  # bhwc
            else:
                x = ff(x) + x  # bhwc
            x = x.permute(0, 3, 1, 2)  # bchw
        return x


class Denoiser(nn.Module):
    """U-Net style encoder-decoder over IGAB blocks, guided by illumination features.

    level=2 gives two downsampling stages, so the channel count runs C -> 2C -> 4C.
    """

    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4], d_state=16,
                 ffn_use_sem=False, use_feature_path=True):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # 3x3 input projection to `dim` channels (F0 in the manuscript figure).
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder: deepen the channels and halve the resolution at each level.
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim

        for i in range(level):  # block count and head count change with the channel depth
            self.encoder_layers.append(nn.ModuleList([
                IGAB(dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim,
                     d_state=d_state, ffn_use_sem=ffn_use_sem, use_feature_path=use_feature_path),
                # FeaDownSample: stride-2 conv, halving H and W and doubling the channels.
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                # IlluFeaDownSample: the same downsampling on the illumination features, so the
                # two stay aligned when they meet again in IFA.
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2
            d_state *= 2

        # Bottleneck at the deepest level (4C).
        self.bottleneck = IGAB(dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1],
                               d_state=d_state, ffn_use_sem=ffn_use_sem,
                               use_feature_path=use_feature_path)

        # Decoder: mirror of the encoder, with a skip connection at every level.
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                # FeaUpSample: transposed conv back to the previous resolution.
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),  # fuses the concatenated skip
                IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                     heads=(dim_level // 2) // dim, d_state=d_state, ffn_use_sem=ffn_use_sem,
                     use_feature_path=use_feature_path)
            ]))
            dim_level //= 2
            d_state //= 2

        # Map back to the output image channels.
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Weight initialisation.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        fea = self.embedding(x)  # F0

        # Encoder pass.
        fea_encoder = []  # skip connections, one per level
        illu_fea_list = []  # illumination features at the matching resolutions
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea, illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck.
        fea = self.bottleneck(fea, illu_fea)

        # Decoder pass.
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea = LeWinBlcok(fea, illu_fea)

        # Global residual.
        out = self.mapping(fea) + x

        return out


class ASANet_Single_Stage(nn.Module):
    """One Retinex stage: illumination estimation followed by guided restoration."""

    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2,
                 num_blocks=[1, 1, 1], d_state=16, illu_use_a=False, illu_use_b=False,
                 ffn_use_sem=False, use_map_path=True, use_feature_path=True):
        super(ASANet_Single_Stage, self).__init__()
        # Fixed M-path gate. Python float => no added parameter/buffer/state_dict key.
        self.map_gate = 1.0 if use_map_path else 0.0
        self.estimator = Illumination_Estimator(
            n_feat, illu_use_a=illu_use_a, illu_use_b=illu_use_b)
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level,
                                 num_blocks=num_blocks, d_state=d_state,
                                 ffn_use_sem=ffn_use_sem,
                                 use_feature_path=use_feature_path)

    def forward(self, img):
        illu_fea, illu_map = self.estimator(img)

        # Two-Path M gate: use_map_path=False strictly recovers input_img = img.
        # The estimator and illu_map remain instantiated/computed in every configuration.
        input_img = img * (self.map_gate * illu_map) + img

        # Restoration, guided by the illumination features.
        output_img = self.denoiser(input_img, illu_fea)

        return output_img


class ASANet(nn.Module):
    """Unified multi-stage model.

    All three component switches off gives the controlled RetinexMamba baseline;
    all three on gives the full ASANet configuration.
    """

    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1, 1, 1],
                 d_state=16, illu_use_a=False, illu_use_b=False, ffn_use_sem=False,
                 use_map_path=True, use_feature_path=True):
        super(ASANet, self).__init__()
        self.stage = stage

        modules_body = [
            ASANet_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2,
                                      num_blocks=num_blocks, d_state=d_state,
                                      illu_use_a=illu_use_a, illu_use_b=illu_use_b,
                                      ffn_use_sem=ffn_use_sem,
                                      use_map_path=use_map_path, use_feature_path=use_feature_path)
            for _ in range(stage)]

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        out = self.body(x)

        return out


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis

    model = ASANet(stage=1, n_feat=40, num_blocks=[1, 2, 2]).cuda()
    print(model)
    inputs = torch.randn((1, 3, 256, 256)).cuda()
    model(inputs)
    flops = FlopCountAnalysis(model, inputs)
    n_param = sum([p.nelement() for p in model.parameters()])  # parameter count
    print(f'GMac:{flops.total() / (1024 * 1024 * 1024)}')
    print(f'Params:{n_param}')
