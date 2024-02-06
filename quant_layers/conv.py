"""
This code is adapted from: https://github.com/hahnyuan/PTQ4ViT)
Copyright (c) 2021.

Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Edward Fish (edward.fish@samsung.com; edward.fish@surrey.ac.uk)
Umberto Michieli (u.michieli@samsung.com)
Mete Ozay (m.ozay@samsung.com)

Licensed under the Creative Commons 
Attribution-NonCommercial-ShareAlike 
4.0 International (CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in 
writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MinMaxQuantConv2d(nn.Conv2d):
    """
    MinMax quantize weight and output
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        mode="raw",
        w_bit=8,
        a_bit=8,
        bias_bit=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.n_calibration_steps = 2
        self.mode = mode
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.bias_bit = bias_bit
        assert bias_bit is None, "No support bias bit now"
        self.w_interval = None
        self.a_interval = None
        self.bias_interval = None
        self.raw_input = None
        self.raw_out = None
        self.metric = None
        self.next_nodes = []
        self.w_qmax = 2 ** (self.w_bit - 1)
        self.a_qmax = 2 ** (self.a_bit - 1)
        self.q_out = []
        self.calibrated = False
        # self.bias_qmax=2**(self.bias_bit-1)

    def forward(self, x):
        """forward method"""

        if self.mode == "raw":
            # self.weight.data = self.weight.data.squeeze(-1)
            out = F.conv1d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )  # pylint: disable=E1102
            # self.weight.data = self.weight.data.unsqueeze(-1)
        elif self.mode == "quant_forward":
            out = self.quant_forward(x)
        elif self.mode == "calibration_step1":
            out = self.calibration_step1(x)
        elif self.mode == "calibration_step2":
            out = self.calibration_step2(x)

        elif self.mode == "sensitivity_step":
            out = self.quant_forward_basic(x)
            # self.q_out.append(out.detach().cpu())
        else:
            raise NotImplementedError
        return out

    def quant_input(self, x):
        """quantize input x"""
        x_sim = (x / self.a_interval).round_().clamp_(-self.a_qmax, self.a_qmax - 1)
        x_sim.mul_(self.a_interval)
        return x_sim

    def calibration_step1(self, x):
        """step1: collection the FP32 values"""
        out = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )  # pylint: disable=E1102
        self.raw_input = x.cpu().detach()
        self.raw_out = out.cpu().detach()
        return out

    def calibration_step2(self, x=0):
        """step2: search for the best S^w and S^a of each layer"""
        self.w_interval = (self.weight.data.abs().max() / (self.w_qmax - 0.5)).detach()
        self.a_interval = (x.abs().max() / (self.a_qmax - 0.5)).detach()
        self.calibrated = True  # pylint disable=w0201
        out = self.quant_forward(x)
        return out


class ChannelwiseBatchingQuantConv2d(MinMaxQuantConv2d):
    """
    Only implemented acceleration with batching_calibration_step2

    setting a_bit to >= 32 will use minmax quantization,
    which means turning off activation quantization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        mode="raw",
        w_bit=8,
        a_bit=8,
        bias_bit=None,
        metric="L2_norm",
        search_round=1,
        eq_alpha=0.1,
        eq_beta=2,
        eq_n=100,
        parallel_eq_n=10,
        n_V=1,
        n_H=1,
        init_layerwise=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            mode=mode,
            w_bit=w_bit,
            a_bit=a_bit,
            bias_bit=bias_bit,
        )
        self.n_V = self.out_channels
        self.n_H = 1
        self.raw_out_s = []
        self.raw_input_s = []
        self.loss_sensitivity = 0
        self.sensitivity_dist = 0
        self.raw_grad_s = []
        self.calib_size = None
        self.calib_batch_size = None
        self.calib_need_batching = None
        self.metric = metric
        self.search_round = search_round
        self.eq_alpha = eq_alpha
        self.eq_beta = eq_beta
        self.eq_n = eq_n
        self.parallel_eq_n = parallel_eq_n
        self.n_H = n_H
        self.n_V = n_V
        self.init_layerwise = init_layerwise
        self.raw_grad = None

    def set_bits(self, bit):
        self.w_bit = bit

    def _initialize_calib_parameters(self):
        """
        set parameters for feeding calibration data
        """
        self.calib_size = self.raw_input.shape[0]
        self.calib_batch_size = self.raw_input.shape[0]
        # self.calib_size = int(self.raw_input.shape[0])
        # self.calib_batch_size = int(self.raw_input.shape[0])
        # self.calib_batch_size = 16
        # self.parralel_eq_n = self.calib_size / self.calib_batch_size
        # self.parallel_eq_n = 16
        while True:
            numel = (
                2
                * (self.raw_input.numel() + self.raw_out.numel())
                / self.calib_size
                * self.calib_batch_size
            )  # number of parameters on GPU
            self.parallel_eq_n = int((2 * 1024 * 1024 * 1024 / 6) // numel)
            if self.parallel_eq_n <= 1:
                self.calib_need_batching = True
                self.calib_batch_size //= 2
            else:
                break

    def _initialize_intervals(self):
        # weight intervals: shape oc,1,1,1
        if self.init_layerwise:
            self.w_interval = (
                ((self.weight.abs().max()) / (self.w_qmax - 0.5))
                .view(1, 1, 1, 1)
                .repeat(self.out_channels, 1, 1, 1)
            )
        else:
            self.weight.data = self.weight.data.unsqueeze(-1)
            self.w_interval = (self.weight.abs().amax([1, 2, 3], keepdim=True)) / (
                self.w_qmax - 0.5
            )

        # activation intervals: shape 1
        tmp_a_intervals = []
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x_ = self.raw_input[b_st:b_ed].cuda()
            a_interval_ = (x_.abs().max() / (self.a_qmax - 0.5)).detach().view(1, 1)
            tmp_a_intervals.append(a_interval_)
        self.a_interval = torch.cat(tmp_a_intervals, dim=1).amax(dim=1, keepdim=False)

    def _get_similarity(self, tensor_raw, tensor_sim, metric=None, raw_grad=None):
        """
        tensor_raw: *, features
        tensor_sim: *, features
        similarity: *, features
        """
        if metric == "cosine":
            # support cosine on patch dim, which is sub-optimal
            # not supporting search best a interval
            b, parallel_eq_n, oc = (
                tensor_sim.shape[0],
                tensor_sim.shape[1],
                tensor_sim.shape[2],
            )
            similarity = F.cosine_similarity(
                tensor_raw.view(b, 1, oc, -1),
                tensor_sim.view(b, parallel_eq_n, oc, -1),
                dim=-1,
            ).view(b, parallel_eq_n, oc, 1, 1)

        elif metric == "pearson":
            # calculate similarity w.r.t complete feature map, but maintain dimension requirement
            b, parallel_eq_n = tensor_sim.shape[0], tensor_sim.shape[1]
            similarity = F.cosine_similarity(
                tensor_raw.view(b, 1, -1), tensor_sim.view(b, parallel_eq_n, -1), dim=-1
            ).view(b, parallel_eq_n, 1, 1)
        else:
            b, eq_n, oc, kw, kh = tensor_sim.shape
            tensor_raw = tensor_raw.view(b, 1, oc, kw, kh)
            if metric == "L1_norm":
                similarity = -torch.abs(tensor_raw - tensor_sim)
            elif metric == "L2_norm":
                similarity = -((tensor_raw - tensor_sim) ** 2)
            elif metric == "linear_weighted_L2_norm":
                similarity = -tensor_raw.abs() * (tensor_raw - tensor_sim) ** 2
            elif metric == "square_weighted_L2_norm":
                similarity = -((tensor_raw * (tensor_raw - tensor_sim)) ** 2)
            elif metric == "hessian":
                assert raw_grad is not None, f"raw_grad is None in _get_similarity!"
                raw_grad = raw_grad.reshape_as(tensor_raw)

                # b, eq_n, oc, kw, kh = tensor_sim.shape
                # tensor_raw = tensor_raw.view(b, 1, oc, kw, kh)
                raw_grad = raw_grad.view(b, 1, oc, kw, kh)

                similarity = -((raw_grad * (tensor_raw - tensor_sim)) ** 2)
            else:
                raise NotImplementedError(f"metric {metric} not implemented!")
        return similarity

    def _search_best_w_interval(self, weight_interval_candidates):
        batch_similarities = []
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_out = self.raw_out[b_st:b_ed].cuda()
            raw_out = raw_out.squeeze()
            raw_grad = self.raw_grad[b_st:b_ed].cuda()
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                # shape: parallel_eq_n,oc,1,1,1
                # [13, 1, 512, 1, 1]
                cur_w_interval = weight_interval_candidates[p_st:p_ed]
                # quantize weight and bias
                # [512, 512, 3, 1] out, in, kw, kh

                if len(self.weight.data.shape) == 3:
                    self.weight.data = self.weight.data.unsqueeze(-1)
                # self.weight.data = self.weight.data.unsqueeze(-1)

                oc, ic, kw, kh = self.weight.data.shape
                # shape: 1,oc,ic,kw,kh #[1, 512, 512, 3, 1]
                w_sim = self.weight.unsqueeze(0)
                w_sim = (
                    (w_sim / cur_w_interval)
                    .round_()
                    .clamp_(-self.w_qmax, self.w_qmax - 1)
                    .mul_(cur_w_interval)
                )  # shape: parallel_eq_n,oc,ic,kw,kh
                # [13, 512, 512, 3, 1] generate 13 candidates
                # shape: parallel_eq_n*oc,ic,kw,kh
                w_sim = w_sim.reshape(-1, ic, kw, kh)  # [6656, 512, 3, 1]
                bias_sim = (
                    self.bias.repeat(p_ed - p_st) if self.bias is not None else None
                )
                # quantize input
                x_sim = self.quant_input(x) if self.a_bit < 32 else x
                # x_sim = [13, 512, 512, 3, 1]

                # calculate similarity and store them
                w_sim = w_sim.squeeze(-1)  # 10240, 512, 3
                x_sim = x_sim.squeeze(-1)  # 4, 512, 32271
                out_sim = F.conv1d(
                    x_sim,
                    w_sim,
                    bias_sim,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )  # shape: b,parallel_eq_n*oc,fw,fh
                # 4, 10240, 16135
                w_sim = w_sim.unsqueeze(-1)
                out_sim = out_sim.unsqueeze(-1)
                out_sim = torch.cat(
                    torch.chunk(out_sim.unsqueeze(1), chunks=p_ed - p_st, dim=2), dim=1
                )  # shape: b,parallel_eq_n,oc,fw,fh
                # possible quantized outputs - choose one after chunk
                # shape: b,parallel_eq_n,oc,fw,fh
                similarity = self._get_similarity(
                    raw_out, out_sim, self.metric, raw_grad
                )
                # shape: b,parallel_eq_n,oc
                similarity = similarity.unsqueeze(-1)
                similarity = torch.mean(similarity, [3, 4])
                # shape: 1, parallel_eq_n, oc
                similarity = torch.sum(similarity, dim=0, keepdim=True)
                similarities.append(similarity)
            # store best weight interval of h into tmp_w_interval
            similarities = torch.cat(similarities, dim=1)  # shape: 1,eq_n,oc
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(
            dim=0, keepdim=False
        )  # shape: eq_n,oc
        best_index = batch_similarities.argmax(dim=0).reshape(
            -1, 1, 1, 1, 1
        )  # shape: 1,oc,1,1,1
        self.w_interval = torch.gather(
            weight_interval_candidates, dim=0, index=best_index
        ).squeeze(dim=0)
        # torch.cuda.empty_cache()

    def _search_best_a_interval(self, input_interval_candidates):
        batch_similarities = []
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_out = self.raw_out[b_st:b_ed].cuda().unsqueeze(1)  # shape: b,1,oc,fw,fh
            raw_grad = self.raw_grad[b_st:b_ed].cuda()
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                # shape: parallel_eq_n,1,1,1,1
                cur_a_interval = input_interval_candidates[p_st:p_ed]

                # quantize weight and bias
                w_sim, bias_sim = self.quant_weight_bias()
                # quantize input
                if len(x.data.shape) == 3:
                    x = x.unsqueeze(-1)
                B, ic, iw, ih = x.shape
                x_sim = x.unsqueeze(0)  # shape: 1,b,ic,iw,ih
                x_sim = (x_sim / (cur_a_interval)).round_().clamp_(
                    -self.a_qmax, self.a_qmax - 1
                ) * (
                    cur_a_interval
                )  # shape: parallel_eq_n,b,ic,iw,ih
                # shape: parallel_eq_n*b,ic,iw,ih
                x_sim = x_sim.view(-1, ic, iw, ih)
                x_sim = x_sim.squeeze(-1)
                # w_sim = w_sim.squeeze(-1)
                # calculate similarity and store them
                out_sim = F.conv1d(
                    x_sim,
                    w_sim,
                    bias_sim,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )  # shape: parallel_eq_n*b,oc,fw,fh
                out_sim = out_sim.unsqueeze(-1)

                out_sim = torch.cat(
                    torch.chunk(out_sim.unsqueeze(0), chunks=p_ed - p_st, dim=1), dim=0
                )  # shape: parallel_eq_n,b,oc,fw,fh
                # shape: b,parallel_eq_n,oc,fw,fh
                out_sim = out_sim.transpose_(0, 1)
                # shape: b,parallel_eq_n,oc,fw,fh
                similarity = self._get_similarity(
                    raw_out, out_sim, self.metric, raw_grad=raw_grad
                )
                # shape: b,parallel_eq_n
                similarity = torch.mean(similarity, dim=[2, 3, 4])
                # shape: 1,parallel_eq_n
                similarity = torch.sum(similarity, dim=0, keepdim=True)
                similarities.append(similarity)
            similarities = torch.cat(similarities, dim=1)  # shape: 1,eq_n
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(
            dim=0, keepdim=False
        )  # shape: eq_n
        a_best_index = batch_similarities.argmax(dim=0).view(1, 1, 1, 1, 1)
        # input_interval_candidates = input_interval_candidates
        self.a_interval = torch.gather(
            input_interval_candidates, dim=0, index=a_best_index
        ).squeeze()

        # self.a_interval = self.a_interval.squeeze()

    def calibration_step2(self, x=0):
        self._initialize_calib_parameters()
        self._initialize_intervals()
        weight_interval_candidates = torch.tensor(
            [
                self.eq_alpha + i * (self.eq_beta - self.eq_alpha) / self.eq_n
                for i in range(self.eq_n + 1)
            ]
        ).cuda().view(-1, 1, 1, 1, 1) * self.w_interval.unsqueeze(
            0
        )  # shape: eq_n,oc,1,1,1
        input_interval_candidates = (
            torch.tensor(
                [
                    self.eq_alpha + i * (self.eq_beta - self.eq_alpha) / self.eq_n
                    for i in range(self.eq_n + 1)
                ]
            )
            .cuda()
            .view(-1, 1, 1, 1, 1)
            * self.a_interval
        )  # shape: eq_n,1,1,1,1
        for e in range(self.search_round):
            # search for best weight interval
            self._search_best_w_interval(weight_interval_candidates)
            # search for best input interval
            if self.a_bit < 32:
                self._search_best_a_interval(input_interval_candidates)
        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad

    def quant_weight_bias(self):
        wi = self.w_interval.squeeze(-1)
        wi = wi.squeeze(-1)
        if len(self.weight.data.shape) == 4:
            self.weight.data = self.weight.data.squeeze(-1)
        w_sim = (
            (self.weight / wi).round_().clamp(-self.w_qmax, self.w_qmax - 1).mul_(wi)
        )

        if len(self.weight.data.shape) == 3:
            self.weight.data = self.weight.data.unsqueeze(-1)
        return w_sim, self.bias

    def quant_forward(self, x):
        # assert self.calibrated is not None,
        # f"You should run calibrate_forward before run quant_forward for {self}"
        w_sim, bias_sim = self.quant_weight_bias()
        x_sim = self.quant_input(x) if self.a_bit < 32 else x
        # w_sim = w_sim.squeeze(2)
        if len(x_sim.shape) == 3:
            x_sim.squeeze(0)
        out = F.conv1d(
            x_sim,
            w_sim,
            bias_sim,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return out

    def quant_forward_basic(self, x):
        """basic quant forward method"""
        x_scale = x.max() / self.a_qmax - 1
        w_scale = self.weight.max() / self.w_qmax - 1

        q_x = (
            (x / x_scale).round_().clamp_(-self.a_qmax, self.a_qmax - 1) * (x_scale)
            if self.a_bit < 32
            else x
        )  # shape: parallel_eq_n,b,ic,iw,ih
        q_w = (self.weight / w_scale).round_().clamp_(-self.w_qmax, self.w_qmax) * (
            w_scale
        )
        out = F.conv1d(
            q_x, q_w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return out
