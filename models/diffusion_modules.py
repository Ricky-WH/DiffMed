from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F

try:
    from diffusers import AutoencoderKL, UNet2DConditionModel
except Exception:  # pragma: no cover
    AutoencoderKL = None
    UNet2DConditionModel = None


class ProjectionHead(nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_dim, kernel_size=1),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, feature_map):
        # feature_map: [B, C, H, W] -> [B, H*W, out_dim]
        projected = self.proj(feature_map).flatten(2).transpose(1, 2)
        return self.norm(projected)


class FrozenConvPyramid(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=2, padding=1),
                nn.GELU(),
            ),
        ])

    def forward(self, images):
        features = []
        hidden = images
        for block in self.blocks:
            hidden = block(hidden)
            features.append(hidden)
        return features


class DiffusionFeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        diffusion_cfg = config.get('diffusion', {})
        self.enabled = diffusion_cfg.get('enabled', False)
        self.hidden_dim = diffusion_cfg.get('hidden_dim', 256)
        self.timestep = diffusion_cfg.get('timestep', 50)
        self.model_id = diffusion_cfg.get('model_id', 'runwayml/stable-diffusion-v1-5')
        self.revision = diffusion_cfg.get('revision')
        self.local_files_only = diffusion_cfg.get('local_files_only', True)
        self.frozen = diffusion_cfg.get('frozen', True)
        self.active_scales = [int(idx) for idx in diffusion_cfg.get('active_scales', [0, 1, 2])]
        self.use_stub = True
        self.vae = None
        self.unet = None
        self._features = {}
        self.register_buffer('pixel_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer('pixel_std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
                             persistent=False)

        channels = [64, 128, 256]
        if self.enabled and AutoencoderKL is not None and UNet2DConditionModel is not None:
            try:
                self.vae = AutoencoderKL.from_pretrained(
                    self.model_id,
                    subfolder='vae',
                    revision=self.revision,
                    local_files_only=self.local_files_only,
                )
                self.unet = UNet2DConditionModel.from_pretrained(
                    self.model_id,
                    subfolder='unet',
                    revision=self.revision,
                    local_files_only=self.local_files_only,
                )
                self._register_hooks()
                channels = [
                    self.unet.config.block_out_channels[0],
                    self.unet.config.block_out_channels[1],
                    self.unet.config.block_out_channels[-1],
                ]
                self.use_stub = False
            except Exception:
                self.use_stub = True

        self.stub_backbone = FrozenConvPyramid(channels)
        self.projections = nn.ModuleList([ProjectionHead(channel, self.hidden_dim) for channel in channels])

        if self.frozen:
            self.freeze_backbone()

    def _register_hooks(self):
        def _save_hook(name):
            def hook(_, __, output):
                value = output[0] if isinstance(output, tuple) else output
                self._features[name] = value
            return hook

        self.unet.down_blocks[0].register_forward_hook(_save_hook('low'))
        self.unet.down_blocks[1].register_forward_hook(_save_hook('mid'))
        self.unet.mid_block.register_forward_hook(_save_hook('high'))

    def freeze_backbone(self):
        modules = [self.stub_backbone] if self.use_stub else [self.vae, self.unet]
        for module in modules:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False

    def set_backbone_trainable(self, trainable=False):
        modules = [self.stub_backbone] if self.use_stub else [self.vae, self.unet]
        for module in modules:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = trainable

    def _denormalize_images(self, images):
        images = images * self.pixel_std + self.pixel_mean
        return images.clamp(0.0, 1.0)

    def _select_active_scales(self, features):
        return [features[idx] for idx in self.active_scales]

    def forward(self, images):
        if not self.enabled:
            return {
                'projected_features': [],
                'raw_features': [],
                'used_stub': True,
            }

        if self.use_stub:
            raw_features = self.stub_backbone(images)
        else:
            self._features = {}
            pixel_values = self._denormalize_images(images)
            pixel_values = pixel_values.mul(2.0).sub(1.0)
            diffusion_dtype = next(self.unet.parameters()).dtype
            latents = self.vae.encode(pixel_values.to(dtype=diffusion_dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            batch_size = latents.size(0)
            timestep = torch.full((batch_size,), self.timestep, device=latents.device, dtype=torch.long)
            encoder_hidden_states = torch.zeros(
                batch_size,
                1,
                self.unet.config.cross_attention_dim,
                device=latents.device,
                dtype=latents.dtype,
            )
            context = torch.no_grad() if self.frozen else nullcontext()
            with context:
                _ = self.unet(latents, timestep, encoder_hidden_states=encoder_hidden_states)
            raw_features = [self._features['low'], self._features['mid'], self._features['high']]

        raw_features = self._select_active_scales(raw_features)
        projected_features = [self.projections[idx](feature.float()) for idx, feature in zip(self.active_scales, raw_features)]
        return {
            'projected_features': projected_features,
            'raw_features': raw_features,
            'used_stub': self.use_stub,
        }


class ScaleGating(nn.Module):
    def __init__(self, text_dim=768, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, question_cls, scale_features):
        # question_cls: [B, H_text], scale_features[i]: [B, N_i, H_diff]
        if len(scale_features) == 0:
            raise ValueError('ScaleGating requires at least one diffusion scale.')

        pooled_features = torch.stack([feature.mean(dim=1) for feature in scale_features], dim=1)
        if pooled_features.size(1) == 1:
            weights = pooled_features.new_ones(pooled_features.size(0), 1)
            return weights, pooled_features, pooled_features[:, 0]

        question_proj = self.text_proj(question_cls)
        logits = []
        for scale_idx in range(pooled_features.size(1)):
            scale_feature = pooled_features[:, scale_idx]
            logits.append(self.score(torch.cat([question_proj, scale_feature], dim=-1)))
        logits = torch.cat(logits, dim=-1)
        weights = torch.softmax(logits, dim=-1)
        fused_feature = (pooled_features * weights.unsqueeze(-1)).sum(dim=1)
        return weights, pooled_features, fused_feature


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query_tokens, key_value_tokens):
        attended, attn_weights = self.cross_attention(
            self.query_norm(query_tokens),
            self.kv_norm(key_value_tokens),
            self.kv_norm(key_value_tokens),
            need_weights=True,
        )
        hidden = query_tokens + attended
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden, attn_weights


class DiffusionAligner(nn.Module):
    def __init__(self, text_dim=768, hidden_dim=256, num_heads=8, num_layers=1, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, text_tokens, diffusion_tokens):
        # text_tokens: [B, L_q, H_text], diffusion_tokens: [B, N_d, H_diff]
        aligned_tokens = self.text_proj(text_tokens)
        attention_maps = []
        for layer in self.layers:
            aligned_tokens, attention_map = layer(aligned_tokens, diffusion_tokens)
            attention_maps.append(attention_map)
        return aligned_tokens, attention_maps


class GatedVisualFusion(nn.Module):
    def __init__(self, vision_dim=768, diffusion_dim=256, dropout=0.1):
        super().__init__()
        self.diffusion_proj = nn.Linear(diffusion_dim, vision_dim)
        self.gate = nn.Sequential(
            nn.Linear(vision_dim * 2, vision_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vision_dim, vision_dim),
        )
        self.output_norm = nn.LayerNorm(vision_dim)

    def forward(self, visual_tokens, diffusion_tokens):
        # visual_tokens: [B, N_v, H_mumc]
        # diffusion_tokens: [B, H_diff] or [B, N_d, H_diff]
        if diffusion_tokens.dim() == 3:
            diffusion_tokens = diffusion_tokens.mean(dim=1)
        diffusion_tokens = self.diffusion_proj(diffusion_tokens).unsqueeze(1).expand(-1, visual_tokens.size(1), -1)
        gate = torch.sigmoid(self.gate(torch.cat([visual_tokens, diffusion_tokens], dim=-1)))
        fused_tokens = self.output_norm(visual_tokens + gate * diffusion_tokens)
        return fused_tokens, gate


def build_weighted_diffusion_bank(scale_features, scale_weights):
    weighted_features = []
    for scale_idx, feature in enumerate(scale_features):
        weighted_features.append(feature * scale_weights[:, scale_idx].view(-1, 1, 1))
    return torch.cat(weighted_features, dim=1)


def info_nce_loss(image_repr, text_repr, temperature=0.07):
    image_repr = F.normalize(image_repr, dim=-1)
    text_repr = F.normalize(text_repr, dim=-1)
    logits = image_repr @ text_repr.t() / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_i2t + loss_t2i)


def scale_decorrelation_loss(scale_features):
    if len(scale_features) == 0:
        raise ValueError('scale_decorrelation_loss requires at least one scale feature.')
    if len(scale_features) < 2:
        return scale_features[0].new_tensor(0.0)

    pooled_features = [F.normalize(feature.mean(dim=1), dim=-1) for feature in scale_features]
    total_loss = pooled_features[0].new_tensor(0.0)
    num_pairs = 0
    for left_idx in range(len(pooled_features)):
        for right_idx in range(left_idx + 1, len(pooled_features)):
            correlation = (pooled_features[left_idx] * pooled_features[right_idx]).sum(dim=-1)
            total_loss = total_loss + correlation.pow(2).mean()
            num_pairs += 1
    return total_loss / max(num_pairs, 1)


def gate_entropy_loss(scale_weights):
    entropy = -(scale_weights * torch.log(scale_weights.clamp_min(1e-8))).sum(dim=-1)
    return entropy.mean()
