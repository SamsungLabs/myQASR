"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Edward Fish (edward.fish@samsung.com; edward.fish@surrey.ac.uk)
Umberto Michieli (u.michieli@samsung.com)
Mete Ozay (m.ozay@samsung.com)

Licensed under the Creative Commons 
Attribution-NonCommercial-ShareAlike 4.0 
International (CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at 
https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, 
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, 
either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

from quant_layers.conv import ChannelwiseBatchingQuantConv2d
from quant_layers.linear import (
    PostGeluPTQSLBatchingQuantLinear,
    PTQSLBatchingQuantLinear,
)
from quant_layers.matmul import PTQSLBatchingQuantMatMul, SoSPTQSLBatchingQuantMatMul

NO_SOFTMAX = False
NO_POST_GELU = False

BIT = 8
conv_fc_name_list = [
    "qconv",
    "qlinear_q",
    "qlinear_k",
    "qlinear_v",
    "qlinear_out",
    "qlinear_MLP_1",
    "qlinear_MLP_2",
    "embedding_layer",
]
matmul_name_list = ["qmatmul_qk", "qmatmul_scorev"]
w_bit = {name: BIT for name in conv_fc_name_list}
a_bit = {name: BIT for name in conv_fc_name_list}
A_bit = {name: BIT for name in matmul_name_list}
B_bit = {name: BIT for name in matmul_name_list}

ptqsl_conv2d_kwargs = {
    "metric": "cosine",
    "eq_alpha": 0.01,
    "eq_beta": 1.2,
    "eq_n": 100,
    "search_round": 3,
    "n_V": 1,
    "n_H": 1,
}
ptqsl_linear_kwargs = {
    "metric": "hessian",
    "eq_alpha": 0.01,
    "eq_beta": 1.2,
    "eq_n": 100,
    "search_round": 3,
    "n_V": 1,
    "n_H": 1,
    "n_a": 1,
    # Conventionally I'll not add an actual bias correction in linear
    "bias_correction": True,
}
ptqsl_matmul_kwargs = {
    "metric": "hessian",
    "eq_alpha": 0.01,
    "eq_beta": 1.2,
    "eq_n": 100,
    "search_round": 3,
    "n_G_A": 1,
    "n_V_A": 1,
    "n_H_A": 1,
    "n_G_B": 1,
    "n_V_B": 1,
    "n_H_B": 1,
}


def get_module(module_type, *args, **kwargs):
    """return module with quantized params"""
    if module_type == "qconv":
        kwargs.update(ptqsl_conv2d_kwargs)

        module = ChannelwiseBatchingQuantConv2d(
            *args, **kwargs, w_bit=w_bit["qconv"], a_bit=8, init_layerwise=False
        )  # turn off activation quantization

    if module_type == "embedding_layer":
        kwargs.update(ptqsl_conv2d_kwargs)
        module = ChannelwiseBatchingQuantConv2d(
            *args, **kwargs, w_bit=8, a_bit=32, init_layerwise=False
        )  # turn off activation quantization
    elif "qlinear" in module_type:
        kwargs.update(ptqsl_linear_kwargs)
        if (
            module_type == "qlinear_q"
            or module_type == "qlinear_k"
            or module_type == "qlinear_v"
        ):
            kwargs["n_V"] = 1  # q, k, v
            module = PTQSLBatchingQuantLinear(
                *args, **kwargs, w_bit=w_bit[module_type], a_bit=a_bit[module_type]
            )
        elif module_type == "qlinear_MLP_2":
            if NO_POST_GELU:
                module = PTQSLBatchingQuantLinear(
                    *args, **kwargs, w_bit=w_bit[module_type], a_bit=a_bit[module_type]
                )
            else:
                module = PostGeluPTQSLBatchingQuantLinear(
                    *args, **kwargs, w_bit=w_bit[module_type], a_bit=a_bit[module_type]
                )
        elif module_type == "qlinear_classifier":
            kwargs["n_V"] = 1
            module = PTQSLBatchingQuantLinear(
                *args, **kwargs, w_bit=w_bit[module_type], a_bit=a_bit[module_type]
            )
        else:
            module = PTQSLBatchingQuantLinear(
                *args, **kwargs, w_bit=w_bit[module_type], a_bit=a_bit[module_type]
            )
    elif "qmatmul" in module_type:
        kwargs.update(ptqsl_matmul_kwargs)
        if module_type == "qmatmul_qk":
            module = PTQSLBatchingQuantMatMul(
                *args, **kwargs, A_bit=A_bit[module_type], B_bit=B_bit[module_type]
            )
        elif module_type == "qmatmul_scorev":
            if NO_SOFTMAX:
                module = PTQSLBatchingQuantMatMul(
                    *args, **kwargs, A_bit=A_bit[module_type], B_bit=B_bit[module_type]
                )
            else:
                module = SoSPTQSLBatchingQuantMatMul(
                    *args, **kwargs, A_bit=A_bit[module_type], B_bit=B_bit[module_type]
                )
    return module
