"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Edward Fish (edward.fish@samsung.com; edward.fish@surrey.ac.uk)
Umberto Michieli (u.michieli@samsung.com)
Mete Ozay (m.ozay@samsung.com)

Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 
4.0 International (CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, 
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""


def calculate_size(bit_depths, parameters):
    """calculate size of model"""
    q_size = (bit_depths / 8) * parameters
    return q_size


def get_model_size(model_dict):
    """get current size"""
    q_size = 0.0
    for name in model_dict.keys():
        q_size += calculate_size(model_dict[name]["bits"], model_dict[name]["size"])
    q_size = bits_to_mb(q_size)
    return q_size


def bits_to_mb(size):
    """convert bits to mb"""
    size = size / 1024**2
    return size
