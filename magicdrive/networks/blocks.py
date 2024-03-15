from typing import Any, Dict, List, Optional, Tuple, Union
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.models.attention import (
    BasicTransformerBlock, AdaLayerNorm, AdaLayerNormZero
)
from diffusers.models.controlnet import zero_module


def _ensure_kv_is_int(view_pair: dict):
    """yaml key can be int, while json cannot. We convert here.
    """
    new_dict = {}
    for k, v in view_pair.items():
        new_value = [int(vi) for vi in v]
        new_dict[int(k)] = new_value
    return new_dict


def get_zero_module(zero_module_type, dim):
    if zero_module_type == "zero_linear":
        # NOTE: zero_module cannot apply to successive layers.
        connector = zero_module(nn.Linear(dim, dim))
    elif zero_module_type == "gated":
        connector = GatedConnector(dim)
    elif zero_module_type == "none":
        # TODO: if this block is in controlnet, we may not need zero here.
        def connector(x): return x
    else:
        raise TypeError(f"Unknown zero module type: {zero_module_type}")
    return connector


class GatedConnector(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        data = torch.zeros(dim)
        self.alpha = nn.parameter.Parameter(data)

    def forward(self, inx):
        # as long as last dim of input == dim, pytorch can auto-broad
        return F.tanh(self.alpha) * inx


class BasicMultiviewTransformerBlock(BasicTransformerBlock):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
        # multi_view
        neighboring_view_pair: Optional[Dict[int, List[int]]] = None,
        neighboring_attn_type: Optional[str] = "add",
        zero_module_type="zero_linear",
    ):
        super().__init__(
            dim, num_attention_heads, attention_head_dim, dropout,
            cross_attention_dim, activation_fn, num_embeds_ada_norm,
            attention_bias, only_cross_attention, double_self_attention,
            upcast_attention, norm_elementwise_affine, norm_type, final_dropout)

        self._args = {
            k: v for k, v in locals().items()
            if k != "self" and not k.startswith("_")}
        self.neighboring_view_pair = _ensure_kv_is_int(neighboring_view_pair)
        self.neighboring_attn_type = neighboring_attn_type
        # multiview attention
        self.norm4 = (
            AdaLayerNorm(dim, num_embeds_ada_norm)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        )
        self.attn4 = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        self.connector = get_zero_module(zero_module_type, dim)

    @property
    def new_module(self):
        ret = {
            "norm4": self.norm4,
            "attn4": self.attn4,
        }
        if isinstance(self.connector, nn.Module):
            ret["connector"] = self.connector
        return ret

    @property
    def n_cam(self):
        return len(self.neighboring_view_pair)

    def _construct_attn_input(self, norm_hidden_states):
        B = len(norm_hidden_states)
        # reshape, key for origin view, value for ref view
        hidden_states_in1 = []
        hidden_states_in2 = []
        cam_order = []
        if self.neighboring_attn_type == "add":
            for key, values in self.neighboring_view_pair.items():
                for value in values:
                    hidden_states_in1.append(norm_hidden_states[:, key])
                    hidden_states_in2.append(norm_hidden_states[:, value])
                    cam_order += [key] * B
            # N*2*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "concat":
            for key, values in self.neighboring_view_pair.items():
                hidden_states_in1.append(norm_hidden_states[:, key])
                hidden_states_in2.append(torch.cat([
                    norm_hidden_states[:, value] for value in values
                ], dim=1))
                cam_order += [key] * B
            # N*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            # N*B, 2*H*W, head*dim
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "self":
            hidden_states_in1 = rearrange(
                norm_hidden_states, "b n l ... -> b (n l) ...")
            hidden_states_in2 = None
            cam_order = None
        else:
            raise NotImplementedError(
                f"Unknown type: {self.neighboring_attn_type}")
        return hidden_states_in1, hidden_states_in2, cam_order

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        attn_output = self.attn1(
            norm_hidden_states, encoder_hidden_states=encoder_hidden_states
            if self.only_cross_attention else None,
            attention_mask=attention_mask, **cross_attention_kwargs,)
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm else self.norm2(hidden_states))
            # TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
            # prepare attention mask here

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # multi-view cross attention
        norm_hidden_states = (
            self.norm4(hidden_states, timestep) if self.use_ada_layer_norm else
            self.norm4(hidden_states)
        )
        # batch dim first, cam dim second
        norm_hidden_states = rearrange(
            norm_hidden_states, '(b n) ... -> b n ...', n=self.n_cam)
        B = len(norm_hidden_states)
        # key is query in attention; value is key-value in attention
        hidden_states_in1, hidden_states_in2, cam_order = self._construct_attn_input(
            norm_hidden_states, )
        # attention
        attn_raw_output = self.attn4(
            hidden_states_in1,
            encoder_hidden_states=hidden_states_in2,
            **cross_attention_kwargs,
        )
        # final output
        if self.neighboring_attn_type == "self":
            attn_output = rearrange(
                attn_raw_output, 'b (n l) ... -> b n l ...', n=self.n_cam)
        else:
            attn_output = torch.zeros_like(norm_hidden_states)
            for cam_i in range(self.n_cam):
                attn_out_mv = rearrange(attn_raw_output[cam_order == cam_i],
                                        '(n b) ... -> b n ...', b=B)
                attn_output[:, cam_i] = torch.sum(attn_out_mv, dim=1)
        attn_output = rearrange(attn_output, 'b n ... -> (b n) ...')
        # apply zero init connector (one layer)
        attn_output = self.connector(attn_output)
        # short-cut
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (
                1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


class LearnablePosEmb(nn.Module):
    def __init__(self, size) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))

    def forward(self, inx):
        return self.param + inx


class BasicTemporalTransformerBlock(BasicTransformerBlock):
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            video_length,  # temporal
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_type: str = "layer_norm",
            final_dropout: bool = False,
            # temporal
            neighboring_view_pair: Optional[Dict[int, List[int]]] = None,
            zero_module_type="zero_linear",
            type="t_last",
            pos_emb="learnable",
    ):
        super().__init__(
            dim, num_attention_heads, attention_head_dim, dropout,
            cross_attention_dim, activation_fn, num_embeds_ada_norm,
            attention_bias, only_cross_attention, double_self_attention,
            upcast_attention, norm_elementwise_affine, norm_type, final_dropout,
        )

        self._args = {
            k: v for k, v in locals().items()
            if k != "self" and not k.startswith("_")}
        self.neighboring_view_pair = _ensure_kv_is_int(neighboring_view_pair)

        # "ts_first_ff", "s_first_t_ff", "s_ff_t_last", "t_first_ff", "t_ff",
        # "t_last" for adding new modules
        # "_ts_first_ff", "_s_first_t_ff", "_s_ff_t_last" for changing self-attn
        self.type = type
        self._video_length = video_length
        # NOTE:
        # 1. the length of sc_attn_index should match video_length
        # 2. each value is a list of indexes, showing the source of kv
        self._sc_attn_index = None

        # SC-Attn
        if self.type in ["ts_first_ff", "s_first_t_ff", "s_ff_t_last"]:
            if self.use_ada_layer_norm:
                self.sc_norm = AdaLayerNorm(dim, num_embeds_ada_norm)
            else:
                self.sc_norm = nn.LayerNorm(
                    dim, elementwise_affine=norm_elementwise_affine)
            self.sc_attn = Attention(
                query_dim=dim,
                cross_attention_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
            self.sc_connector = get_zero_module(zero_module_type, dim)

        # temporal attn
        if pos_emb == "learnable":
            self.pos_emb = LearnablePosEmb(size=(1, self.video_length, dim))
        elif pos_emb == "none":
            self.pos_emb = None
        else:
            raise NotImplementedError(f"Unknown type {pos_emb}")

        if self.use_ada_layer_norm:
            self.temp_norm = AdaLayerNorm(dim, num_embeds_ada_norm)
        else:
            self.temp_norm = nn.LayerNorm(
                dim, elementwise_affine=norm_elementwise_affine)
        self.temp_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        self.temp_connector = get_zero_module(zero_module_type, dim)

    @property
    def sc_attn_index(self):
        # one can set `self._sc_attn_index` to a function for convenient changes
        # among batches.
        if callable(self._sc_attn_index):
            return self._sc_attn_index()
        else:
            return self._sc_attn_index

    @property
    def n_cam(self):
        return len(self.neighboring_view_pair)

    @property
    def video_length(self):
        if self.sc_attn_index is not None:
            assert len(self.sc_attn_index) == self._video_length
        return self._video_length

    @property
    def new_module(self):
        ret = {
            "temp_attn": self.temp_attn,
            "temp_norm": self.temp_norm,
        }
        if isinstance(self.temp_connector, nn.Module):
            ret["temp_connector"] = self.temp_connector
        if isinstance(self.pos_emb, nn.Module):
            ret["temp_pos_emb"] = self.pos_emb
        if self.type in ["ts_first_ff", "s_first_t_ff", "s_ff_t_last"]:
            ret["sc_attn"] = self.sc_attn
            ret["sc_norm"] = self.sc_norm
            if isinstance(self.sc_connector, nn.Module):
                ret["sc_connector"] = self.sc_connector
        if self.type.startswith("_"):
            ret['attn1.to_q'] = self.attn1.to_q
        return ret

    def forward_old_attns(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        shift_mlp, scale_mlp, gate_mlp = None, None, None
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        if self.type.startswith("_"):
            norm_hidden_states = rearrange(
                norm_hidden_states, "(b f n) d c -> (b n) f d c",
                f=self.video_length, n=self.n_cam)
            B = len(norm_hidden_states)

            # this index is for kv pair, your dataloader should make it consistent.
            norm_hidden_states_q, norm_hidden_states_kv, back_order = self._construct_sc_attn_input(
                norm_hidden_states, self.sc_attn_index, type="concat")

            attn_raw_output = self.attn1(
                norm_hidden_states_q,
                encoder_hidden_states=norm_hidden_states_kv,
                attention_mask=attention_mask, **cross_attention_kwargs)
            attn_output = torch.zeros_like(norm_hidden_states)
            for frame_i in range(self.video_length):
                # TODO: any problem here? n should == 1
                attn_out_mt = rearrange(attn_raw_output[back_order == frame_i],
                                        '(n b) ... -> b n ...', b=B)
                attn_output[:, frame_i] = torch.sum(attn_out_mt, dim=1)
            attn_output = rearrange(
                attn_output, "(b n) f d c -> (b f n) d c", n=self.n_cam)
        else:
            attn_output = self.attn1(
                norm_hidden_states, encoder_hidden_states=encoder_hidden_states
                if self.only_cross_attention else None,
                attention_mask=attention_mask, **cross_attention_kwargs,)
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm else self.norm2(hidden_states))
            # TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
            # prepare attention mask here

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        return hidden_states, shift_mlp, scale_mlp, gate_mlp

    def _construct_sc_attn_input(
        self, norm_hidden_states, sc_attn_index, type="add"
    ):
        # assume data has form (b, frame, c), frame == len(sc_attn_index)
        # return two sets of hidden_states and an order list.
        B = len(norm_hidden_states)
        hidden_states_in1 = []
        hidden_states_in2 = []
        back_order = []
        if type == "add":
            for key, values in zip(range(len(sc_attn_index)), sc_attn_index):
                for value in values:
                    hidden_states_in1.append(norm_hidden_states[:, key])
                    hidden_states_in2.append(norm_hidden_states[:, value])
                    back_order += [key] * B
            # N*2*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            back_order = torch.LongTensor(back_order)
        elif type == "concat":
            for key, values in zip(range(len(sc_attn_index)), sc_attn_index):
                hidden_states_in1.append(norm_hidden_states[:, key])
                hidden_states_in2.append(torch.cat([
                    norm_hidden_states[:, value] for value in values
                ], dim=1))
                back_order += [key] * B
            # N*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            # N*B, 3*H*W, head*dim
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            back_order = torch.LongTensor(back_order)
        else:
            raise NotImplementedError(f"Unknown type: {type}")
        return hidden_states_in1, hidden_states_in2, back_order

    def forward_sc_attn(self, hidden_states, timestep):
        # How to we know a frame is a keyframe?
        # No need, we use `self.sc_attn_index` here. Dataloader should maintain
        # the attention relationship.
        hidden_states_in = rearrange(
            hidden_states, "(b f n) d c -> (b n) f d c", f=self.video_length,
            n=self.n_cam)

        norm_hidden_states = (
            self.temp_attn(hidden_states_in, timestep)
            if self.use_ada_layer_norm else self.temp_norm(hidden_states_in))
        B = len(norm_hidden_states)

        # this index is for kv pair, your dataloader should make it consistent.
        norm_hidden_states_q, norm_hidden_states_kv, back_order = self._construct_sc_attn_input(
            norm_hidden_states, self.sc_attn_index)

        attn_raw_output = self.sc_attn(
            norm_hidden_states_q, encoder_hidden_states=norm_hidden_states_kv)
        attn_output = torch.zeros_like(norm_hidden_states)
        for frame_i in range(self.video_length):
            # TODO: any problem here? n should == 3
            attn_out_mt = rearrange(attn_raw_output[back_order == frame_i],
                                    '(n b) ... -> b n ...', b=B)
            attn_output[:, frame_i] = torch.sum(attn_out_mt, dim=1)

        # apply zero init connector (one layer)
        attn_output = self.sc_connector(attn_output)
        attn_output = rearrange(
            attn_output, "(b n) f d c -> (b f n) d c", n=self.n_cam)
        # short-cut
        hidden_states = attn_output + hidden_states
        return hidden_states

    def add_temp_pos_emb(self, hidden_state):
        if self.pos_emb is None:
            return hidden_state
        return self.pos_emb(hidden_state)

    def forward_temporal(self, hidden_states, timestep):
        # Temporal-Attention
        d = hidden_states.shape[1]
        hidden_states_in = rearrange(
            hidden_states, "(b f n) d c -> (b d n) f c", f=self.video_length,
            n=self.n_cam)
        hidden_states_in = self.add_temp_pos_emb(hidden_states_in)
        norm_hidden_states = (
            self.temp_norm(hidden_states_in, timestep)
            if self.use_ada_layer_norm else self.temp_norm(hidden_states_in))
        # NOTE: xformers cannot take bs larger than 8192
        if len(norm_hidden_states) >= 8192:
            chunk_num = math.ceil(len(norm_hidden_states) / 4096.)
            norm_hidden_states = norm_hidden_states.chunk(chunk_num)
            attn_output = torch.cat([
                self.temp_attn(norm_hidden_states[i]) for i in range(chunk_num)
            ], dim=0)
        else:
            attn_output = self.temp_attn(norm_hidden_states)
        # apply zero init connector (one layer)
        attn_output = self.temp_connector(attn_output)
        attn_output = rearrange(
            attn_output, "(b d n) f c -> (b f n) d c", d=d, n=self.n_cam)
        # short-cut
        hidden_states = attn_output + hidden_states
        return hidden_states

    def forward_ff_last(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        if self.type in ["ts_first_ff", "t_first_ff", "_ts_first_ff"]:
            # forward temporal
            hidden_states = self.forward_temporal(hidden_states, timestep)

        # forward sc_attn
        if self.type in ["ts_first_ff", "s_first_t_ff"]:
            hidden_states = self.forward_sc_attn(hidden_states, timestep)

        # 1-2. old attention modules
        # we make first dim like (b f n), it should be safe for old attns to
        # handle it as (b n)
        hidden_states, shift_mlp, scale_mlp, gate_mlp = self.forward_old_attns(
            hidden_states, attention_mask, encoder_hidden_states,
            encoder_attention_mask, timestep, cross_attention_kwargs,
            class_labels,
        )

        if self.type in ["s_first_t_ff", "t_ff", "_s_first_t_ff"]:
            # forward temporal here
            hidden_states = self.forward_temporal(hidden_states, timestep)

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (
                1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states
        return hidden_states

    def forward_t_last(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        # forward sc_attn
        if self.type in ["s_ff_t_last"]:
            hidden_states = self.forward_sc_attn(hidden_states, timestep)

        # 1-2. old attention modules
        # we make first dim like (b f n), it should be safe for old attns to
        # handle it as (b n)
        hidden_states, shift_mlp, scale_mlp, gate_mlp = self.forward_old_attns(
            hidden_states, attention_mask, encoder_hidden_states,
            encoder_attention_mask, timestep, cross_attention_kwargs,
            class_labels,
        )

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (
                1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        # temporal
        hidden_states = self.forward_temporal(hidden_states, timestep)
        return hidden_states

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        if self.type.endswith("_ff"):
            hidden_states = self.forward_ff_last(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
                cross_attention_kwargs,
                class_labels,
            )
        elif self.type.endswith("t_last"):
            hidden_states = self.forward_t_last(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
                cross_attention_kwargs,
                class_labels,
            )
        else:
            raise NotImplementedError(f"Unknown type {self.type}")

        return hidden_states


class TemporalMultiviewTransformerBlock(BasicTemporalTransformerBlock):
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            video_length,  # temporal
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_type: str = "layer_norm",
            final_dropout: bool = False,
            # multi_view
            neighboring_view_pair: Optional[Dict[int, List[int]]] = None,
            neighboring_attn_type: Optional[str] = "add",
            zero_module_type="zero_linear",
            # temporal
            type="t_last",
            pos_emb="learnable",
            zero_module_type2="zero_linear",
            spatial_trainable=False,
    ):
        super().__init__(
            dim, num_attention_heads, attention_head_dim, video_length,
            dropout, cross_attention_dim, activation_fn, num_embeds_ada_norm,
            attention_bias, only_cross_attention, double_self_attention,
            upcast_attention, norm_elementwise_affine, norm_type, final_dropout,
            neighboring_view_pair, zero_module_type2, type, pos_emb)

        self._args = {
            k: v for k, v in locals().items()
            if k != "self" and not k.startswith("_")}
        self.spatial_trainable = spatial_trainable

        self.neighboring_attn_type = neighboring_attn_type
        # multiview attention
        self.norm4 = (
            AdaLayerNorm(dim, num_embeds_ada_norm)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        )
        self.attn4 = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        self.connector = get_zero_module(zero_module_type, dim)

    def _construct_attn_input(self, norm_hidden_states):
        B = len(norm_hidden_states)
        # reshape, key for origin view, value for ref view
        hidden_states_in1 = []
        hidden_states_in2 = []
        cam_order = []
        if self.neighboring_attn_type == "add":
            for key, values in self.neighboring_view_pair.items():
                for value in values:
                    hidden_states_in1.append(norm_hidden_states[:, key])
                    hidden_states_in2.append(norm_hidden_states[:, value])
                    cam_order += [key] * B
            # N*2*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "concat":
            for key, values in self.neighboring_view_pair.items():
                hidden_states_in1.append(norm_hidden_states[:, key])
                hidden_states_in2.append(torch.cat([
                    norm_hidden_states[:, value] for value in values
                ], dim=1))
                cam_order += [key] * B
            # N*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            # N*B, 2*H*W, head*dim
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "self":
            hidden_states_in1 = rearrange(
                norm_hidden_states, "b n l ... -> b (n l) ...")
            hidden_states_in2 = None
            cam_order = None
        else:
            raise NotImplementedError(
                f"Unknown type: {self.neighboring_attn_type}")
        return hidden_states_in1, hidden_states_in2, cam_order

    @property
    def new_module(self):
        if self.spatial_trainable:
            ret = {
                "norm4": self.norm4,
                "attn4": self.attn4,
            }
            if isinstance(self.connector, nn.Module):
                ret["connector"] = self.connector
        else:
            ret = {}
        ret = {
            "temp_attn": self.temp_attn,
            "temp_norm": self.temp_norm,
            **ret,
        }
        if isinstance(self.temp_connector, nn.Module):
            ret["temp_connector"] = self.temp_connector
        if isinstance(self.pos_emb, nn.Module):
            ret["temp_pos_emb"] = self.pos_emb
        if self.type in ["ts_first_ff", "s_first_t_ff", "s_ff_t_last"]:
            ret["sc_attn"] = self.sc_attn
            ret["sc_norm"] = self.sc_norm
            if isinstance(self.sc_connector, nn.Module):
                ret["sc_connector"] = self.sc_connector
        if self.type.startswith("_"):
            ret['attn1.to_q'] = self.attn1.to_q
        return ret

    def forward_old_attns(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        shift_mlp, scale_mlp, gate_mlp = None, None, None
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        if self.type.startswith("_"):
            norm_hidden_states = rearrange(
                norm_hidden_states, "(b f n) d c -> (b n) f d c",
                f=self.video_length, n=self.n_cam)
            B = len(norm_hidden_states)

            # this index is for kv pair, your dataloader should make it consistent.
            norm_hidden_states_q, norm_hidden_states_kv, back_order = self._construct_sc_attn_input(
                norm_hidden_states, self.sc_attn_index, type="concat")

            attn_raw_output = self.attn1(
                norm_hidden_states_q,
                encoder_hidden_states=norm_hidden_states_kv,
                attention_mask=attention_mask, **cross_attention_kwargs)
            attn_output = torch.zeros_like(norm_hidden_states)
            for frame_i in range(self.video_length):
                # TODO: any problem here? n should == 1
                attn_out_mt = rearrange(attn_raw_output[back_order == frame_i],
                                        '(n b) ... -> b n ...', b=B)
                attn_output[:, frame_i] = torch.sum(attn_out_mt, dim=1)
            attn_output = rearrange(
                attn_output, "(b n) f d c -> (b f n) d c", n=self.n_cam)
        else:
            attn_output = self.attn1(
                norm_hidden_states, encoder_hidden_states=encoder_hidden_states
                if self.only_cross_attention else None,
                attention_mask=attention_mask, **cross_attention_kwargs,)
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm else self.norm2(hidden_states))
            # TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
            # prepare attention mask here

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # multi-view cross attention
        norm_hidden_states = (
            self.norm4(hidden_states, timestep) if self.use_ada_layer_norm else
            self.norm4(hidden_states)
        )
        # batch dim first, cam dim second
        norm_hidden_states = rearrange(
            norm_hidden_states, '(b n) ... -> b n ...', n=self.n_cam)
        B = len(norm_hidden_states)
        # key is query in attention; value is key-value in attention
        hidden_states_in1, hidden_states_in2, cam_order = self._construct_attn_input(
            norm_hidden_states, )
        # attention
        attn_raw_output = self.attn4(
            hidden_states_in1,
            encoder_hidden_states=hidden_states_in2,
            **cross_attention_kwargs,
        )
        # final output
        if self.neighboring_attn_type == "self":
            attn_output = rearrange(
                attn_raw_output, 'b (n l) ... -> b n l ...', n=self.n_cam)
        else:
            attn_output = torch.zeros_like(norm_hidden_states)
            for cam_i in range(self.n_cam):
                attn_out_mv = rearrange(attn_raw_output[cam_order == cam_i],
                                        '(n b) ... -> b n ...', b=B)
                attn_output[:, cam_i] = torch.sum(attn_out_mv, dim=1)
        attn_output = rearrange(attn_output, 'b n ... -> (b n) ...')
        # apply zero init connector (one layer)
        attn_output = self.connector(attn_output)
        # short-cut
        hidden_states = attn_output + hidden_states
        return hidden_states, shift_mlp, scale_mlp, gate_mlp

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        if self.type.endswith("_ff"):
            hidden_states = self.forward_ff_last(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
                cross_attention_kwargs,
                class_labels,
            )
        elif self.type.endswith("t_last"):
            hidden_states = self.forward_t_last(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
                cross_attention_kwargs,
                class_labels,
            )
        else:
            raise NotImplementedError(f"Unknown type {self.type}")

        return hidden_states
