from functools import partial

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .diffusion_modules import (
    DiffusionAligner,
    DiffusionFeatureEncoder,
    GatedVisualFusion,
    ScaleGating,
    build_weighted_diffusion_bank,
    gate_entropy_loss,
    info_nce_loss,
    scale_decorrelation_loss,
)
from .vision.vit import VisionTransformer
from .xbert import BertConfig, BertLMHeadModel, BertModel


def normalize_model_variant(variant):
    alias = {
        'mumc': 'mumc',
        'baseline': 'mumc',
        'diffrep': 'mumc_diffrep',
        'mumc_diffrep': 'mumc_diffrep',
        'diffalign': 'mumc_diffalign',
        'mumc_diffalign': 'mumc_diffalign',
        'diffrepalign': 'mumc_diffrepalign',
        'mumc_diffrepalign': 'mumc_diffrepalign',
        'mumc_diffalignrep': 'mumc_diffrepalign',
    }
    return alias.get(str(variant).lower(), 'mumc')


class MUMC_VQA(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 text_decoder=None,
                 tokenizer=None,
                 config=None,
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.config = config
        self.distill = config['distill']
        self.model_variant = normalize_model_variant(config.get('model_variant', 'mumc'))
        self.use_diffrep = config.get('use_diffrep', self.model_variant in ['mumc_diffrep', 'mumc_diffrepalign'])
        self.use_diffalign = config.get('use_diffalign', self.model_variant in ['mumc_diffalign', 'mumc_diffrepalign'])
        self.use_diffusion = self.use_diffrep or self.use_diffalign
        self.use_global_contrast = config.get('use_global_contrast', self.use_diffusion)
        self.use_answer_contrast = config.get('use_answer_contrast', False)
        self.use_decorrelation = config.get('use_decorrelation', self.use_diffusion)
        self.use_gate_entropy = config.get('use_gate_entropy', False)

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        config_encoder = BertConfig.from_json_file(config['bert_config'])
        config_decoder = BertConfig.from_json_file(config['bert_config'])
        self.hidden_size = config_encoder.hidden_size

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=config_encoder, add_pooling_layer=False)

        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layers = 6
        self.text_decoder = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)

        self.diffusion_cfg = config.get('diffusion', {})
        self.loss_cfg = config.get('loss', {})
        self.optimizer_cfg = config.get('optimizer', {})
        self.train_stage = str(config.get('train_stage', 'baseline')).lower()

        if self.use_diffusion:
            align_dim = self.diffusion_cfg.get('hidden_dim', 256)
            dropout = self.diffusion_cfg.get('dropout', 0.1)
            self.diffusion_encoder = DiffusionFeatureEncoder(config)
            self.scale_gating = ScaleGating(text_dim=self.hidden_size, hidden_dim=align_dim, dropout=dropout)
            self.diffusion_aligner = DiffusionAligner(
                text_dim=self.hidden_size,
                hidden_dim=align_dim,
                num_heads=self.diffusion_cfg.get('num_heads', 8),
                num_layers=self.diffusion_cfg.get('num_layers', 1),
                dropout=dropout,
            )
            self.diffrep_fusion = GatedVisualFusion(self.hidden_size, align_dim, dropout=dropout)
            self.diffalign_fusion = GatedVisualFusion(self.hidden_size, align_dim, dropout=dropout)
            self.global_image_proj = nn.Linear(self.hidden_size, align_dim)
            self.global_text_proj = nn.Linear(self.hidden_size, align_dim)
            if self.use_answer_contrast:
                self.answer_joint_proj = nn.Linear(self.hidden_size, align_dim)
                self.answer_text_proj = nn.Linear(self.hidden_size, align_dim)

        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=config_encoder,
                                                            add_pooling_layer=False)
            self.text_decoder_m = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)
            self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                                [self.text_encoder, self.text_encoder_m],
                                [self.text_decoder, self.text_decoder_m],
                                ]
            self.copy_params()
            self.momentum = 0.995

        self.configure_training_stage(self.train_stage)

    def configure_training_stage(self, stage='baseline'):
        self.train_stage = str(stage).lower()
        if not self.use_diffusion:
            return

        self._set_trainable(self.visual_encoder, True)
        self._set_trainable(self.text_encoder, True)
        self._set_trainable(self.text_decoder, True)
        self.diffusion_encoder.freeze_backbone()
        self._set_new_modules_trainable(True)

        if self.train_stage == 'diffusion':
            self._set_trainable(self.visual_encoder, False)
            self._set_trainable(self.text_encoder, False)
            self._set_trainable(self.text_decoder, True)
            self._unfreeze_visual_tail(self.config.get('unfreeze_visual_last_n_blocks', 0))
            self._unfreeze_text_tail(self.config.get('unfreeze_text_last_n_layers', 0))
        elif self.train_stage == 'joint':
            self._set_trainable(self.visual_encoder, False)
            self._set_trainable(self.text_encoder, False)
            self._set_trainable(self.text_decoder, True)
            self._unfreeze_visual_tail(self.config.get('unfreeze_visual_last_n_blocks', 2))
            self._unfreeze_text_tail(self.config.get('unfreeze_text_last_n_layers', 2))

        if not self.diffusion_cfg.get('frozen', True):
            self.diffusion_encoder.set_backbone_trainable(True)

    def get_optimizer_groups(self, config):
        init_lr = float(config.get('init_lr', 2e-5))
        min_lr = float(config.get('min_lr', 1e-8))
        if not self.use_diffusion:
            return [{
                'params': [param for param in self.parameters() if param.requires_grad],
                'lr': init_lr,
                'init_lr': init_lr,
                'min_lr': min_lr,
                'weight_decay': float(config.get('weight_decay', 0.05)),
            }]

        backbone_lr = float(self.optimizer_cfg.get('backbone_lr', 1e-5))
        new_lr = float(self.optimizer_cfg.get('new_lr', 1e-4))
        decoder_lr = float(self.optimizer_cfg.get('decoder_lr', new_lr))
        diffusion_backbone_lr = float(self.optimizer_cfg.get('diffusion_backbone_lr', 5e-6))
        weight_decay = float(self.optimizer_cfg.get('weight_decay', config.get('weight_decay', 0.05)))

        parameter_groups = {
            'backbone': {'params': [], 'lr': backbone_lr, 'init_lr': backbone_lr, 'min_lr': min_lr, 'weight_decay': weight_decay},
            'decoder': {'params': [], 'lr': decoder_lr, 'init_lr': decoder_lr, 'min_lr': min_lr, 'weight_decay': weight_decay},
            'new_modules': {'params': [], 'lr': new_lr, 'init_lr': new_lr, 'min_lr': min_lr, 'weight_decay': weight_decay},
            'diffusion_backbone': {
                'params': [],
                'lr': diffusion_backbone_lr,
                'init_lr': diffusion_backbone_lr,
                'min_lr': min_lr,
                'weight_decay': weight_decay,
            },
        }

        new_prefixes = [
            'diffusion_encoder.projections',
            'scale_gating',
            'diffusion_aligner',
            'diffrep_fusion',
            'diffalign_fusion',
            'global_image_proj',
            'global_text_proj',
            'answer_joint_proj',
            'answer_text_proj',
        ]
        diffusion_backbone_prefixes = [
            'diffusion_encoder.vae',
            'diffusion_encoder.unet',
            'diffusion_encoder.stub_backbone',
        ]

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('text_decoder'):
                parameter_groups['decoder']['params'].append(param)
            elif any(name.startswith(prefix) for prefix in new_prefixes):
                parameter_groups['new_modules']['params'].append(param)
            elif any(name.startswith(prefix) for prefix in diffusion_backbone_prefixes):
                parameter_groups['diffusion_backbone']['params'].append(param)
            else:
                parameter_groups['backbone']['params'].append(param)

        return [group for group in parameter_groups.values() if len(group['params']) > 0]

    def _set_new_modules_trainable(self, is_trainable):
        for module_name in ['scale_gating', 'diffusion_aligner', 'diffrep_fusion', 'diffalign_fusion',
                            'global_image_proj', 'global_text_proj', 'answer_joint_proj', 'answer_text_proj']:
            module = getattr(self, module_name, None)
            if module is None:
                continue
            self._set_trainable(module, is_trainable)

    @staticmethod
    def _set_trainable(module, is_trainable):
        for param in module.parameters():
            param.requires_grad = is_trainable

    def _unfreeze_visual_tail(self, num_blocks):
        if num_blocks <= 0:
            return
        for block in self.visual_encoder.blocks[-num_blocks:]:
            self._set_trainable(block, True)
        self.visual_encoder.cls_token.requires_grad = True
        self.visual_encoder.pos_embed.requires_grad = True
        self._set_trainable(self.visual_encoder.norm, True)

    def _unfreeze_text_tail(self, num_layers):
        if num_layers <= 0:
            return
        for layer in self.text_encoder.encoder.layer[-num_layers:]:
            self._set_trainable(layer, True)

    def _encode_question_text(self, question_inputs):
        return self.text_encoder(question_inputs.input_ids,
                                 attention_mask=question_inputs.attention_mask,
                                 return_dict=True,
                                 mode='text')

    def _apply_diffusion_branch(self, image, image_embeds, question_inputs):
        text_outputs = self._encode_question_text(question_inputs)
        question_tokens = text_outputs.last_hidden_state
        question_cls = question_tokens[:, 0, :]
        diffusion_outputs = self.diffusion_encoder(image)
        scale_features = diffusion_outputs['projected_features']
        scale_weights, pooled_scales, weighted_global = self.scale_gating(question_cls, scale_features)

        fused_image_embeds = image_embeds
        artifacts = {
            'scale_weights': scale_weights,
            'pooled_scales': pooled_scales,
            'used_stub': diffusion_outputs['used_stub'],
            'align_attention': [],
            'diffrep_gate': None,
            'diffalign_gate': None,
        }

        if self.use_diffrep:
            fused_image_embeds, diffrep_gate = self.diffrep_fusion(fused_image_embeds, weighted_global)
            artifacts['diffrep_gate'] = diffrep_gate

        token_bank = build_weighted_diffusion_bank(scale_features, scale_weights)
        aligned_tokens = None
        if self.use_diffalign:
            aligned_tokens, attention_maps = self.diffusion_aligner(question_tokens, token_bank)
            fused_image_embeds, diffalign_gate = self.diffalign_fusion(fused_image_embeds, aligned_tokens)
            artifacts['align_attention'] = attention_maps
            artifacts['diffalign_gate'] = diffalign_gate

        return {
            'question_tokens': question_tokens,
            'question_cls': question_cls,
            'fused_image_embeds': fused_image_embeds,
            'token_bank': token_bank,
            'aligned_tokens': aligned_tokens,
            'scale_features': scale_features,
            'artifacts': artifacts,
        }

    def _compute_auxiliary_losses(self, answer_inputs, question_output, diffusion_state):
        losses = {}
        if not self.use_diffusion:
            return losses

        temperature = float(self.loss_cfg.get('contrastive_temperature', 0.07))
        if self.use_global_contrast:
            image_repr = self.global_image_proj(diffusion_state['fused_image_embeds'][:, 0, :])
            text_repr = self.global_text_proj(diffusion_state['question_cls'])
            losses['global_contrast'] = info_nce_loss(image_repr, text_repr, temperature=temperature)

        if self.use_answer_contrast:
            answer_text_output = self.text_encoder(answer_inputs.input_ids,
                                                   attention_mask=answer_inputs.attention_mask,
                                                   return_dict=True,
                                                   mode='text')
            joint_repr = self.answer_joint_proj(question_output.last_hidden_state[:, 0, :])
            answer_repr = self.answer_text_proj(answer_text_output.last_hidden_state[:, 0, :])
            losses['answer_contrast'] = info_nce_loss(joint_repr, answer_repr, temperature=temperature)

        if self.use_decorrelation:
            losses['decorrelation'] = scale_decorrelation_loss(diffusion_state['scale_features'])

        if self.use_gate_entropy:
            losses['gate_entropy'] = gate_entropy_loss(diffusion_state['artifacts']['scale_weights'])

        return losses

    def _combine_losses(self, vqa_loss, auxiliary_losses):
        total_loss = vqa_loss
        if 'global_contrast' in auxiliary_losses:
            total_loss = total_loss + float(self.loss_cfg.get('lambda_global_contrast', 0.1)) * auxiliary_losses['global_contrast']
        if 'answer_contrast' in auxiliary_losses:
            total_loss = total_loss + float(self.loss_cfg.get('lambda_answer_contrast', 0.0)) * auxiliary_losses['answer_contrast']
        if 'decorrelation' in auxiliary_losses:
            total_loss = total_loss + float(self.loss_cfg.get('lambda_decorrelation', 0.05)) * auxiliary_losses['decorrelation']
        if 'gate_entropy' in auxiliary_losses:
            total_loss = total_loss + float(self.loss_cfg.get('lambda_gate_entropy', 0.0)) * auxiliary_losses['gate_entropy']
        return total_loss

    def forward(self, image, question, answer=None, alpha=0, k=None, train=True):
        # image_embeds: [B, N_v, H_mumc]
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        question_inputs = self.tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(image.device)
        answer_inputs = self.tokenizer(answer, padding='longest', return_tensors="pt").to(image.device)

        diffusion_state = None
        fused_image_embeds = image_embeds
        if self.use_diffusion:
            diffusion_state = self._apply_diffusion_branch(image, image_embeds, question_inputs)
            fused_image_embeds = diffusion_state['fused_image_embeds']

        if train:
            answer_targets = answer_inputs.input_ids.masked_fill(answer_inputs.input_ids == self.tokenizer.pad_token_id, -100)

            question_output = self.text_encoder(question_inputs.input_ids,
                                                attention_mask=question_inputs.attention_mask,
                                                encoder_hidden_states=fused_image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)
            if self.distill:
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image)
                    question_output_m = self.text_encoder_m(question_inputs.input_ids,
                                                            attention_mask=question_inputs.attention_mask,
                                                            encoder_hidden_states=image_embeds_m,
                                                            encoder_attention_mask=image_atts,
                                                            return_dict=True)

                    logits_m = self.text_decoder_m(answer_inputs.input_ids,
                                                   attention_mask=answer_inputs.attention_mask,
                                                   encoder_hidden_states=question_output_m.last_hidden_state,
                                                   encoder_attention_mask=question_inputs.attention_mask,
                                                   return_logits=True,
                                                   )

                answer_output = self.text_decoder(answer_inputs.input_ids,
                                                  attention_mask=answer_inputs.attention_mask,
                                                  encoder_hidden_states=question_output.last_hidden_state,
                                                  encoder_attention_mask=question_inputs.attention_mask,
                                                  labels=answer_targets,
                                                  return_dict=True,
                                                  soft_labels=F.softmax(logits_m, dim=-1),
                                                  alpha=alpha,
                                                  reduction='none',
                                                  )
            else:
                answer_output = self.text_decoder(answer_inputs.input_ids,
                                                  attention_mask=answer_inputs.attention_mask,
                                                  encoder_hidden_states=question_output.last_hidden_state,
                                                  encoder_attention_mask=question_inputs.attention_mask,
                                                  labels=answer_targets,
                                                  return_dict=True,
                                                  reduction='none',
                                                  )

            vqa_loss = answer_output.loss.sum() / image.size(0)
            auxiliary_losses = self._compute_auxiliary_losses(answer_inputs, question_output, diffusion_state)
            total_loss = self._combine_losses(vqa_loss, auxiliary_losses)

            loss_dict = {
                'loss': total_loss,
                'vqa_loss': vqa_loss,
            }
            loss_dict.update(auxiliary_losses)
            return loss_dict

        question_output = self.text_encoder(question_inputs.input_ids,
                                            attention_mask=question_inputs.attention_mask,
                                            encoder_hidden_states=fused_image_embeds,
                                            encoder_attention_mask=image_atts,
                                            return_dict=True)
        topk_ids, topk_probs = self.rank_answer(question_output.last_hidden_state, question_inputs.attention_mask,
                                                answer_inputs.input_ids, answer_inputs.attention_mask, k)

        return topk_ids, topk_probs

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):

        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)

        start_output = self.text_decoder(start_ids,
                                         encoder_hidden_states=question_states,
                                         encoder_attention_mask=question_atts,
                                         return_dict=True,
                                         reduction='none')
        logits = start_output.logits[:, 0, :]

        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)

        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        input_ids = []
        input_atts = []
        for _, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))

        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id,
                                            -100)

        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.text_decoder(input_ids,
                                   attention_mask=input_atts,
                                   encoder_hidden_states=question_states,
                                   encoder_attention_mask=question_atts,
                                   labels=targets_ids,
                                   return_dict=True,
                                   reduction='none')

        answer_loss = output.loss
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)
        topk_probs = F.softmax(log_probs_sum, dim=-1)

        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)
        return topk_ids, topk_probs


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))
