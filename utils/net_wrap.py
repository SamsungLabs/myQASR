"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Edward Fish (edward.fish@samsung.com; edward.fish@surrey.ac.uk)
Umberto Michieli (u.michieli@samsung.com)
Mete Ozay (m.ozay@samsung.com)

Licensed under the Creative Commons 
Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, 
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import torch
import torch.nn as nn


class MatMul(nn.Module):
    """matrix multiplication with torch dot operator"""

    def forward(self, A, B):
        return A @ B


def _fold_bn(conv_module, bn_module):
    """fold batch norm layers into weight"""
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def fold_bn_into_conv(conv_module, bn_module):
    """fold batch norm into conv layers"""
    w, b = _fold_bn(conv_module, bn_module)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b.data)
    else:
        conv_module.bias.data = b.data
    conv_module.weight.data = w.data


def wrap_modules_in_net(net, cfg, uniform=None):
    """wrap each module in the network with correct key for quantization"""

    wrapped_modules = {}
    bit_depths = {}
    module_dict = {}
    module_types = {
        "query": "qlinear_q",
        "linear_q": "qlinear_q",
        "linear_k": "qlinear_k",
        "linear_v": "qlinear_v",
        "key": "qlinear_k",
        "value": "qlinear_v",
        "qkv": "qlinear_qkv",
        "proj": "qlinear_proj",
        "fc1": "qlinear_MLP_1",
        "fc2": "qlinear_MLP_2",
        "head": "qlinear_classifier",
        "matmul1": "qmatmul_qk",
        "matmul2": "qmatmul_scorev",
        "reduction": "qlinear_reduction",
        "q_proj": "qlinear_q",
        "v_proj": "qlinear_v",
        "k_proj": "qlinear_k",
        "linear_pos": "qlinear_out",
        "linear_out": "qlinear_out",
        "out_proj": "qlinear_out",
        "out": "qlinear_out",
        "0": "qlinear_MLP_1",
        "2": "qlinear_MLP_2",
        "projector": "qlinear_out",
        "projection": "qlinear_out",
        "classifier": "qlinear_out",
        "intermediate_dense": "qlinear_MLP_1",
        "output_dense": "qlinear_MLP_2",
    }
    # "wav2vec2.feature_extractor.conv_layers.0.conv"]
    # "wav2vec2.feature_extractor.conv_layers.0.conv"]
    skip_modules = [
        "lm_head",
        "wav2vec2.encoder.pos_conv_embed.conv",
        "wav2vec2_conformer.encoder.pos_conv_embed.conv",
    ]

    it = [(name, m) for name, m in net.named_modules()]

    for name, m in it:
        # if name in skip_modules:
        #     print(name)
        #     continue

        module_dict[name] = m
        if name in skip_modules:
            continue

        idx = name.rfind(".")
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")

        if isinstance(m, nn.Conv1d):
            idx = idx + 1 if idx != 0 else idx
            if (
                name == "wav2vec2.feature_extractor.conv_layers.0.conv"
                or name == "encoder.conv1"
            ):
                layer_name = "embedding_layer"
            else:
                layer_name = "qconv"

            new_m = cfg.get_module(
                layer_name,
                m.in_channels,
                m.out_channels,
                m.kernel_size,
                m.stride,
                m.padding,
                m.dilation,
                m.groups,
                m.bias is not None,
                m.padding_mode,
            )
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            replace_m = new_m
            wrapped_modules[name] = new_m
            bit_depths[name] = new_m.w_bit
            setattr(father_module, name[idx:], replace_m)
        elif isinstance(m, nn.Linear):
            # Linear Layer
            if name == "lm_head" or name == "proj_out":
                continue
            idx = idx + 1 if idx != 0 else idx
            new_m = cfg.get_module(
                module_types[name[idx:]], m.in_features, m.out_features
            )
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            replace_m = new_m
            wrapped_modules[name] = new_m
            bit_depths[name] = new_m.w_bit
            setattr(father_module, name[idx:], replace_m)
        elif isinstance(m, MatMul):
            # if "decoder" in name:
            #     continue
            # Matmul Layer
            idx = idx + 1 if idx != 0 else idx
            new_m = cfg.get_module(module_types[name[idx:]])
            replace_m = new_m
            wrapped_modules[name] = new_m
            # bit_depths[name] = new_m.w_bit
            setattr(father_module, name[idx:], replace_m)
    print("Completed net wrap.")
    return wrapped_modules, bit_depths
