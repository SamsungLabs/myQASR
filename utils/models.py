"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Edward Fish (edward.fish@samsung.com; edward.fish@surrey.ac.uk)
Umberto Michieli (u.michieli@samsung.com)
Mete Ozay (m.ozay@samsung.com)

Licensed under the Creative Commons 
Attribution-NonCommercial-ShareAlike 4.0 International 
(CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at 
https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, 
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

from types import MethodType
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import WindowAttention
from timm.models.vision_transformer import Attention
from whisper import MultiHeadAttention


def attention_forward(self, x):
    """replace forward attn with matmul operation"""
    B, N, C = x.shape
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    # make torchscript happy (cannot use tensor as tuple)
    q, k, v = qkv.unbind(0)
    # q = [32, 12, 197, 64]

    # attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    del q, k

    # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    del attn, v
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def window_attention_forward(self, x, mask=None):
    """replace window attn forward with matmul"""
    B_, N, C = x.shape
    qkv = (
        self.qkv(x)
        .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    # make torchscript happy (cannot use tensor as tuple)
    q, k, v = qkv.unbind(0)

    q = q * self.scale
    # attn = (q @ k.transpose(-2, -1))
    attn = self.matmul1(q, k.transpose(-2, -1))

    relative_position_bias = self.relative_position_bias_table[
        self.relative_position_index.view(-1)
    ].view(
        self.window_size[0] * self.window_size[1],
        self.window_size[0] * self.window_size[1],
        -1,
    )  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(
        2, 0, 1
    ).contiguous()  # nH, Wh*Ww, Wh*Ww
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
            1
        ).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)

    attn = self.attn_drop(attn)

    # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def whisper_att_forward(self, q, k, v, mask=None):
    """replace whisper att forward with decomposed matmul"""
    _, n_ctx, n_state = q.shape
    scale = (n_state // self.n_head) ** -0.25
    q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
    k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
    v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

    qk = self.matmul1(q, k)
    if mask is not None:
        qk = qk + mask[:n_ctx, :n_ctx]
    qk = qk.float()

    w = F.softmax(qk, dim=-1).to(q.dtype)
    return self.matmul2(w, v).permute(0, 2, 1, 3).flatten(start_dim=2)


def wav2vec_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None
    bsz, tgt_len, embed_dim = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states) * self.scaling
    # get key, value proj
    if is_cross_attention and past_key_value is not None:
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        # cross_attentions
        key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
        value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    elif past_key_value is not None:
        # reuse k, v, self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor)
        # of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder)
        # save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states.
        # Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current
        # projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_states, value_states)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    query_states = query_states.view(bsz, self.num_heads, -1, self.head_dim)
    key_states = key_states.view(bsz, self.num_heads, -1, self.head_dim)

    src_len = key_states.size(2)
    attn_weights = self.matmul1(query_states, key_states.transpose(-2, -1))
    # print(attn_weights.shape)
    attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            )
        attn_weights = (
            attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        )
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if layer_head_mask is not None:
        if layer_head_mask.size() != (self.num_heads,):
            raise ValueError(
                f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            )
        attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
            bsz, self.num_heads, tgt_len, src_len
        )
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if output_attentions:
        # this operation is a bit awkward, but it's required to
        # make sure that attn_weights keeps its gradient.
        # In order to do so, attn_weights have to be reshaped
        # twice and have to be reused in the following
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(
            bsz * self.num_heads, tgt_len, src_len
        )
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(
        attn_weights, p=self.dropout, training=self.training
    )
    attn_probs = attn_probs.view(bsz, self.num_heads, tgt_len, src_len)
    value_states = value_states.view(bsz, self.num_heads, -1, self.head_dim)
    attn_output = self.matmul2(attn_probs, value_states)
    attn_output = attn_output.view(bsz * self.num_heads, tgt_len, self.head_dim)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights_reshaped, past_key_value


class MatMul(nn.Module):
    """new matmul using dot-product torch operator"""

    def forward(self, A, B):
        """forward via new matmul"""
        return A @ B


def get_net(net):
    """
    Get a vision transformer model.
    This will replace matrix multiplication operations with matmul modules in the model.

    Currently support almost all models in timm.models.transformers, including:
    - vit_tiny/small/base/large_patch16/patch32_224/384,
    - deit_tiny/small/base(_distilled)_patch16_224,
    - deit_base(_distilled)_patch16_384,
    - swin_tiny/small/base/large_patch4_window7_224,
    - swin_base/large_patch4_window12_384

    These models are finetuned on imagenet-1k and should use ViTImageNetLoaderGenerator
    for calibration and testing.
    """
    # if "wav2vec" in name or "whisper" in name or "Wav2Vec" in name or "Whisper" in name:
    #     if "wav2vec" in name:
    #         net = Wav2Vec2ForCTC.from_pretrained(f"facebook/{name}-960h")
    #     elif "whisper" in name or "Whisper" in name:
    #         net = whisper.load_model("base")

    # else:
    #     net = timm.create_model(name, pretrained=True)
    #     # weights = torch.load(
    #     #     '/home/CORP/edward.fish/SR_personalization/SRUK/Compression/PTQ4ViT/weights/vit-base-cutmix-20epochs.pth')
    #     # net.load_state_dict(weights['state_dict'])

    for _, module in net.named_modules():
        for _, module in module.named_modules():
            if isinstance(module, Attention):
                setattr(module, "matmul1", MatMul())
                setattr(module, "matmul2", MatMul())
                module.forward = MethodType(attention_forward, module)
            if isinstance(module, WindowAttention):
                setattr(module, "matmul1", MatMul())
                setattr(module, "matmul2", MatMul())
                module.forward = MethodType(window_attention_forward, module)
            if isinstance(module, MultiHeadAttention):
                setattr(module, "matmul1", MatMul())
                setattr(module, "matmul2", MatMul())
                module.forward = MethodType(wav2vec_forward, module)

    net.cuda()
    net.eval()
    return net
