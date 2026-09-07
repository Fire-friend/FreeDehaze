from typing import Optional, Callable
import numpy as np
import torch
import xformers.ops


from diffusers.models.attention_processor import Attention

import torch.nn.functional as F
from priors import adaptive_instance_normalization_qkv, modified_sigmoid_adjusted


def classify_blocks(block_list, name):
    is_correct_block = False
    for block in block_list:
        if block in name:
            is_correct_block = True
            break
    return is_correct_block


class MySelfAttnProcessor:
    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            (batch_size, channel, height, width) = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        (batch_size, key_tokens, _) = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, key_tokens, batch_size
        )
        self.attention_mask = attention_mask
        self.attn = attn
        if attention_mask is not None:
            (_, query_tokens, _) = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        self.key = key
        self.query = query
        self.value = value
        self.hidden_state = hidden_states.detach()
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()
        hidden_states = _attention(query, key, value, attention_mask, attn.scale)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class MySelfAttnProcessor2:
    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

    def contrast_attn(self, attn_map, contrast_factor):
        attn_mean = torch.mean(attn_map, dim=1, keepdim=True)
        attn_map = (attn_map - attn_mean) * contrast_factor + attn_mean
        attn_map = torch.clip(attn_map, min=0.0, max=1.0)
        return attn_map

    def contrast_attn2(self, attn_map, contrast_factor):
        attn_mean = torch.mean(attn_map, dim=0, keepdim=True)
        attn_map = (attn_map - attn_mean) * contrast_factor + attn_mean
        attn_map = torch.clip(attn_map, min=0.0, max=1.0)
        return attn_map

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            (batch_size, channel, height, width) = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        (batch_size, key_tokens, _) = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, key_tokens, batch_size
        )
        self.attention_mask = attention_mask
        self.attn = attn
        if attention_mask is not None:
            (_, query_tokens, _) = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )
        query = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if not is_cross and self.injection_schedule is not None:
            source_batch_size = int(query.shape[0] // 5)
            q_source = query[2 * source_batch_size : 3 * source_batch_size].clone()
            k_source = key[2 * source_batch_size : 3 * source_batch_size].clone()
            v_source1 = value[4 * source_batch_size : 5 * source_batch_size].clone()
            s = int(
                np.sqrt(
                    self.DCP_mask.shape[2] * self.DCP_mask.shape[3] / q_source.shape[1]
                )
            )
            if self.t in self.injection_schedule or self.t == 1000:
                if self.weight_inject != 0:
                    DCP_mask = F.interpolate(self.DCP_mask, scale_factor=1 / s).reshape(
                        1, -1, 1
                    )
                    DCP_mask = modified_sigmoid_adjusted(DCP_mask, a=self.weight_inject)
                    # The sky mask must be taken before sigmoid modulation.
                    sky = (
                        F.interpolate(self.DCP_mask, scale_factor=1 / s).reshape(
                            1, -1, 1
                        )
                        == 0.1
                    )
                    non_sky = DCP_mask[~sky]
                    DCP_mask = torch.where(
                        sky,
                        non_sky.mean()
                        if non_sky.numel()
                        else torch.zeros((), device=DCP_mask.device),
                        DCP_mask,
                    )
                    query[3 * source_batch_size : 4 * source_batch_size] = (
                        DCP_mask * q_source
                        + (1 - DCP_mask)
                        * query[3 * source_batch_size : 4 * source_batch_size]
                    )
                    query[0 * source_batch_size : 1 * source_batch_size] = (
                        DCP_mask * q_source
                        + (1 - DCP_mask)
                        * query[0 * source_batch_size : 1 * source_batch_size]
                    )
                    key[3 * source_batch_size : 4 * source_batch_size] = (
                        DCP_mask * k_source
                        + (1 - DCP_mask)
                        * key[3 * source_batch_size : 4 * source_batch_size]
                    )
                    key[0 * source_batch_size : 1 * source_batch_size] = (
                        DCP_mask * k_source
                        + (1 - DCP_mask)
                        * key[0 * source_batch_size : 1 * source_batch_size]
                    )
                else:
                    query[3 * source_batch_size : 4 * source_batch_size] = q_source
                    query[0 * source_batch_size : 1 * source_batch_size] = q_source
                    key[3 * source_batch_size : 4 * source_batch_size] = k_source
                    key[0 * source_batch_size : 1 * source_batch_size] = k_source
            if self.t > self.inject_app_stop_t:
                query[4 * source_batch_size : 5 * source_batch_size] = q_source + 0.1
                query[1 * source_batch_size : 2 * source_batch_size] = q_source + 0.1
                key[4 * source_batch_size : 5 * source_batch_size] = k_source + 0.1
                key[1 * source_batch_size : 2 * source_batch_size] = k_source + 0.1
            else:
                temp = value[3 * source_batch_size : 4 * source_batch_size].clone()
                temp = adaptive_instance_normalization_qkv(temp, v_source1)
                value[3 * source_batch_size : 4 * source_batch_size] = temp
        self.key = key
        self.query = query
        self.value = value
        self.hidden_state = hidden_states.detach()
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()
        if self.my_att == False:
            hidden_states = _attention(query, key, value, attention_mask, attn.scale)
        else:
            sim = torch.einsum("b i d, b j d -> b i j", query, key) * attn.scale
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(attn.heads, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)
            at_prob = sim.softmax(dim=-1)
            (dd, h, w) = at_prob.shape
            at_prob = at_prob.reshape(batch_size, -1, h, w)
            att_temp = self.contrast_attn(at_prob[[0, 3]], 1.67)
            at_prob = torch.cat(
                [att_temp[0:1], at_prob[[1, 2]], att_temp[1:2], at_prob[4:5]], dim=0
            ).reshape(-1, h, w)
            hidden_states = torch.einsum("b i j, b j d -> b i d", at_prob, value)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def prep_unet_attention(unet, config, injection_schedule=None, DCP_mask=None):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention":
            module.set_processor(MySelfAttnProcessor())
    if injection_schedule is not None:
        res_dict = config.guidance.cross_attn.res_dict
        for res in res_dict:
            for block in res_dict[res]:
                module = (
                    unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
                )
                module.set_processor(MySelfAttnProcessor2())
                if res == 3 or res == 2:
                    setattr(module.processor, "my_att", False)
                else:
                    setattr(module.processor, "my_att", False)
                setattr(module.processor, "injection_schedule", injection_schedule)
                setattr(module.processor, "DCP_mask", DCP_mask)
                setattr(
                    module.processor,
                    "weight_inject",
                    config.guidance.cross_attn.weight_inject,
                )
                setattr(
                    module.processor,
                    "inject_app_stop_t",
                    config.guidance.cross_attn.inject_app_stop_t,
                )
    return unet


def _attention(query, key, value, mask, scale):
    # Match the attention backend used by run.py during denoising.
    return xformers.ops.memory_efficient_attention(
        query,
        key,
        value,
        attn_bias=mask,
        scale=scale,
    )


def register_time(pipeline, timestep):
    for module in pipeline.unet.modules():
        if hasattr(module, "processor"):
            module.processor.t = timestep
        if hasattr(module, "injection_schedule"):
            module.t = timestep


def clear_features(unet):
    for module in unet.modules():
        processor = getattr(module, "processor", None)
        for name in ("key", "query", "value", "hidden_state"):
            if processor is not None and hasattr(processor, name):
                setattr(processor, name, None)
        if hasattr(module, "record_hidden_state"):
            module.record_hidden_state = None
