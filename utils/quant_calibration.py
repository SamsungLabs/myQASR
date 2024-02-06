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
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, 
either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import numpy as np
import torch
import torch.nn.functional as F

from scipy.stats import normaltest

from sklearn.preprocessing import minmax_scale
from tqdm import tqdm
from quant_layers.conv import MinMaxQuantConv2d
from quant_layers.linear import MinMaxQuantLinear
from quant_layers.matmul import MinMaxQuantMatMul
from utils.model_est import get_model_size


def grad_hook(module, grad_input, grad_output):
    """add gradient hooks"""
    if module.raw_grad is None:
        module.raw_grad = []
    module.raw_grad.append(grad_output[0].cpu().detach())  # that's a tuple!


def linear_forward_hook(module, d_input, output):
    """add linear hooks on forward pass"""
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(d_input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())


def conv2d_forward_hook(module, d_input, output):
    """add hooks on convolutions"""
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(d_input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())


def matmul_forward_hook(module, d_input, output):
    """add matrix multiplication hooks"""
    if module.raw_input is None:
        module.raw_input = [[], []]
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input[0].append(d_input[0].cpu().detach())
    module.raw_input[1].append(d_input[1].cpu().detach())
    module.raw_out.append(output.cpu().detach())


class CustomQuantCalibrator:
    """
    Modularization of hessian_quant_calib

    Hessian metric needs gradients of layer outputs to weigh the loss,
    which calls for back propagation in calibration, both sequentially
    and parallelly. Despite the complexity of bp, hessian quant calibrator
    is compatible with other non-gradient quantization metrics.
    """

    def __init__(
        self, net, wrapped_modules, calib_loader, sequential=False, batch_size=1
    ):
        self.net = net
        self.wrapped_modules = wrapped_modules
        self.calib_loader = calib_loader
        self.sequential = sequential
        self.calibrated = False
        self.batch_size = batch_size
        self.global_model_dict, self.model_size = self.set_model_get_size()

        print("init model size", self.model_size, "mb")

    def set_sensitivities(self, args, sensitivity_dict, name, module, grad=False):
        """set sensitivity method for distances"""
        out = module.raw_out
        if args.sensitivity_method == "normal":
            out = out.flatten()
            k, p = normaltest(out)
            sensitivity_dict[name]["out"] = k

        if args.sensitivity_method == "max":
            # sensitivity_dict[name]["in"] = module.raw_input.max()
            sensitivity_dict[name]["out"] = out.max()
        elif args.sensitivity_method == "abs_max":
            sensitivity_dict[name]["out"] = torch.abs(out).max()
            #  sensitivity_dict[name]["in"] = torch.abs(module.raw_input).max()
        elif args.sensitivity_method == "std":
            sensitivity_dict[name]["out"] = torch.abs(out).std()
            # sensitivity_dict[name]["in"] = torch.abs(module.raw_input).std()
        elif args.sensitivity_method == "var":
            sensitivity_dict[name]["out"] = torch.abs(out).var()
            # sensitivity_dict[name]["in"] = torch.abs(module.raw_input).var()
        elif args.sensitivity_method == "median":
            sensitivity_dict[name]["out"] = torch.abs(out).median()
            # sensitivity_dict[name]["in"] = torch.abs(module.raw_input).median()
        elif args.sensitivity_method == "mean":
            sensitivity_dict[name]["out"] = torch.abs(out).mean()
            # sensitivity_dict[name]["in"] = torch.abs(module.raw_input).mean()
        if grad:
            grad_out = torch.quantile(module.raw_grad, q=args.percentile, dim=-1)
            if args.sensitivity_method == "grad_max":
                sensitivity_dict[name]["out"] = torch.abs(grad_out).max()
            elif args.sensitivity_method == "grad_min":
                sensitivity_dict[name]["out"] = grad_out.min()
        return sensitivity_dict

    def set_uniform_values(self, args, calib_layers):
        """set uniform values"""
        for name, module in calib_layers:
            module.w_bit = args.w_bit
            module.w_qmax = 2 ** (module.w_bit - 1)
            module.a_bit = args.a_bit

        for name, m in self.net.named_parameters():
            name = name.replace(".weight", "")
            self.global_model_dict[name] = {"bits": args.w_bit, "size": m.nelement()}
        return self.global_model_dict, get_model_size(self.global_model_dict)

    def set_model_get_size(self):
        """init model at fp and get size"""
        global_model_dict = {}
        for name, m in self.net.named_parameters():
            name = name.replace(".weight", "")
            global_model_dict[name] = {"bits": 32, "size": m.nelement()}
        return global_model_dict, get_model_size(global_model_dict)

    def register_hook(self, module):
        """register hooks"""
        hooks = []
        if isinstance(module, MinMaxQuantLinear):
            hooks.append(module.register_forward_hook(linear_forward_hook))
        if isinstance(module, MinMaxQuantConv2d):
            hooks.append(module.register_forward_hook(conv2d_forward_hook))
        if isinstance(module, MinMaxQuantMatMul):
            hooks.append(module.register_forward_hook(matmul_forward_hook))
        if hasattr(module, "metric"):
            hooks.append(module.register_full_backward_hook(grad_hook))
        return hooks

    def get_soft_preds(self, args):
        """get softmax predictions for distance metrics"""
        soft_preds = []
        with torch.no_grad():
            for inp, _ in self.calib_loader:
                soft_preds = self.net(inp.cuda()).logits
                soft_preds = F.softmax(soft_preds, dim=-1).detach()
                del inp
        return soft_preds

    def process_module_for_sensitivity(
        self, name, module, q, sensitivity_dict, soft_preds, args
    ):
        """get metrics with hooks"""
        for name, module in q:
            q.set_postfix_str(name)
            sensitivity_dict[name] = {}
            hooks = self.register_hook(module)

            self.feed_calibration_data(module, soft_preds, args)

            if isinstance(
                module, (MinMaxQuantLinear, MinMaxQuantConv2d, MinMaxQuantMatMul)
            ):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
                sensitivity_dict = self.set_sensitivities(
                    args, sensitivity_dict, name, module
                )

            if hasattr(module, "metric"):
                module.metric = args.calibration_method
                module.raw_grad = torch.cat(module.raw_grad, dim=0)
                sensitivity_dict = self.set_sensitivities(
                    args, sensitivity_dict, name, module, grad=True
                )

            for hook in hooks:
                hook.remove()
        return sensitivity_dict

    def feed_calibration_data(self, module, soft_preds, args):
        """calibrate quantizer with calib data"""
        for inp, _ in self.calib_loader:
            for batch_st in range(0, self.calib_loader.batch_size, self.batch_size):
                self.net.zero_grad()
                inp_ = inp[batch_st : batch_st + self.batch_size].cuda()
                if inp_.size()[0] == 0:  # weird side effect bug
                    return
                pred = self.net(inp_).logits
                loss = F.kl_div(
                    F.log_softmax(pred, dim=-1),
                    soft_preds[batch_st : batch_st + self.batch_size],
                    reduction="batchmean",
                )
                loss.backward()
                del inp_, pred, loss
                torch.cuda.empty_cache()

    def compute_distance(self, args, raw_preds_flat, q_preds, loss):
        """check distance metric"""
        if args.sensitivity_method == "kl":
            return torch.abs(torch.mean(loss, dim=-1)).cpu()
        if args.sensitivity_method == "dist_l1":
            return torch.abs(
                torch.mean(-torch.abs(raw_preds_flat - q_preds), dim=-1)
            ).cpu()
        if args.sensitivity_method == "dist_l2":
            return torch.abs(
                torch.mean(-(raw_preds_flat * (raw_preds_flat - q_preds) ** 2), dim=-1)
            ).cpu()
        if args.sensitivity_method == "dist_frob":
            return torch.abs(
                torch.mean(-(torch.norm(raw_preds_flat) - torch.norm(q_preds)), dim=-1)
            ).cpu()
        if args.sensitivity_method == "dist_spec":
            return torch.abs(
                torch.mean(
                    -(
                        torch.norm(raw_preds_flat.unsqueeze(0), p="nuc")
                        - torch.norm(q_preds.unsqueeze(0), p="nuc")
                    ),
                    dim=-1,
                )
            ).cpu()

    def calculate_sensitivity_dist(self, args, q, sensitivity_dict, soft_preds):
        """check sensitivity of layer to quantization"""
        for name, module in q:
            with torch.no_grad():
                for inp, target in self.calib_loader:
                    module.mode = "sensitivity_step"
                    q_preds = self.net(inp.cuda()).logits
                    qlog_preds = F.log_softmax(q_preds, dim=-1)
                    loss = F.kl_div(qlog_preds, soft_preds, reduction="batchmean")
                    break  # Remove if you want to loop through all data

                raw_preds_flat = soft_preds.flatten()
                q_preds = q_preds.detach().flatten()

                dist = self.compute_distance(args, raw_preds_flat, q_preds, loss)
                sensitivity_dict[name]["out"] = dist

                module.mode = "raw"
                del q_preds, qlog_preds
                torch.cuda.empty_cache()
        return sensitivity_dict

    def select_bits(self, sensitivity_list, calib_layers, name_list, args):
        """seelct bits for each layer based on sensitivity"""
        print("Model size", self.model_size)
        model_size = self.model_size
        indexes = np.argsort(sensitivity_list)
        if args.sorting_method == "uniform_reverse":
            indexes = indexes[::-1]
        elif args.sorting_method == "shuffle":
            np.random.shuffle(indexes)
        elif args.sorting_method == "min_max":
            if not args.min_bit or not args.max_bit:
                raise Exception("must set min and max bit for min_max_pooling method")
            bits = minmax_scale(indexes, (args.min_bit, args.max_bit))
            for i in indexes:
                name, module = calib_layers[i]
                module.w_bit = round(bits[i])
                module.w_qmax = 2 ** (module.w_bit - 1)
                self.global_model_dict[name]["bits"] = module.w_bit
            bits = [round(bit) for bit in bits]
            model_size = get_model_size(self.global_model_dict)

        # linear constraint method described in algorithm 1 of the paper
        bits = [32] * len(name_list)
        while model_size > args.target:
            for i in indexes:
                name, module = calib_layers[i]
                if not "mat" in name:
                    self.global_model_dict[name]["bits"] -= 1
                    bits[i] -= 1
                    module.w_bit = self.global_model_dict[name]["bits"]
                    module.w_qmax = 2 ** (module.w_bit - 1)
                    model_size = get_model_size(self.global_model_dict)
                    if model_size < args.target:
                        break
                else:
                    bits[i] = 8

        return bits, model_size

    def batching_quant_calib(self, args):
        calib_layers = []
        for name, module in self.wrapped_modules.items():
            calib_layers.append((name, module))

        if args.uniform:
            # no need to perform sensititvity analysis
            self.global_model_dict, self.model_size = self.set_uniform_values(
                args, calib_layers
            )
        else:
            sensitivity_dict = {}
            q = tqdm(
                self.wrapped_modules.items(), desc="tracking values for calibration"
            )

            soft_preds = self.get_soft_preds(args)
            sensitivity_dict = self.process_module_for_sensitivity(
                name, module, q, sensitivity_dict, soft_preds, args
            )
            if args.sensitivity_method in [
                "kl",
                "dist_l1",
                "dist_l2",
                "dist_frob",
                "dist_spec",
            ]:
                sensitivity_dict = self.calculate_sensitivity_dist(
                    args, q, sensitivity_dict, soft_preds
                )

            # Create list of values and names to be sorted later
            name_list = []
            sensitivity_list = []

            for name, module in calib_layers:
                q.set_postfix_str(name)
                sensitivity_list.append(sensitivity_dict[name]["out"])
                name_list.append(name)
                module.a_bit = args.a_bit
                module.a_qmax = 2 ** (module.a_bit - 1)

            bits, model_size = self.select_bits(
                sensitivity_list, calib_layers, name_list, args
            )
            avg_bits = sum(bits) / len(bits)
            min_bits = min(bits)
            max_bits = max(bits)
            std_bits = np.std(bits)

        compressed_size = model_size
        print("compressed model size", compressed_size)

        q = tqdm(self.wrapped_modules.items(), desc="Calibration")

        for name, module in q:
            q.set_postfix_str(name)

            # run calibration step2
            with torch.no_grad():
                if isinstance(module, MinMaxQuantLinear):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantConv2d):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantMatMul):
                    module.calibration_step2()

                torch.cuda.empty_cache()
                # # finishing up current module calibration
                if self.sequential:
                    module.mode = "quant_forward"
                else:
                    module.mode = "raw"

        # finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"

        print("hessian calibration finished")
        if args.uniform:
            bits = [args.w_bit]

        return (
            avg_bits,
            min_bits,
            max_bits,
            std_bits,
            compressed_size,
            args.a_bit,
            self.global_model_dict,
            0,
            0,
            bits,
        )
