"""
This code is adapted from: https://github.com/hahnyuan/PTQ4ViT)
Copyright (c) 2021.

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

from itertools import product

import torch
from torch import nn
from torch.nn import functional as F


class MinMaxQuantMatMul(nn.Module):
    """Matrix Multiplication base class"""

    def __init__(self, A_bit=8, B_bit=8, mode="raw"):
        super().__init__()
        self.A_bit = A_bit
        self.B_bit = B_bit
        self.A_interval = None
        self.B_interval = None
        self.A_qmax = 2 ** (self.A_bit - 1)
        self.B_qmax = 2 ** (self.B_bit - 1)
        self.mode = mode
        self.raw_input = None
        self.raw_out = None

    def forward(self, A, B):
        """main forward method"""
        if self.mode == "raw":
            out = A @ B
        elif self.mode == "quant_forward":
            out = self.quant_forward(A, B)
        elif self.mode == "calibration_step1":
            out = self.calibration_step1(A, B)
        elif self.mode == "calibration_step2":
            out = self.calibration_step2(A, B)
        else:
            raise NotImplementedError
        return out

    def quant_input(self, x, interval, qmax):
        """quantize input"""
        x_sim = (x / interval).round_().clamp_(-qmax, qmax - 1)
        x_sim.mul_(interval)
        return x_sim

    def quant_forward(self, A, B):
        """forward quantize weights"""
        assert (
            self.calibrated is not None
        ), f"You should run calibrate_forward before run quant_forward for {self}"
        A_sim = self.quant_input(A, self.A_interval, self.A_qmax)
        B_sim = self.quant_input(B, self.B_interval, self.B_qmax)
        out = A_sim @ B_sim
        return out

    def calibration_step1(self, A, B):
        """collect raw input values"""
        # step1: collection the FP32 values
        self.raw_input = A.cpu().detach(), B.cpu().detach()
        out = A @ B
        self.raw_out = out.cpu().detach()
        return out


class PTQSLQuantMatMul(MinMaxQuantMatMul):
    """
    Chunk matrix into blockes and quantize.
    Chunking follows naive padding strategy.
    Alternately search for best intervals of each individual blocks for A and B.

    two different scenarios:
    - Q @ K:
        - A's shape: B,H,S,W
        - B's shape: B,H,W,S
    - scores @ V:
        - A's shape: B,H,S,S
        - B's shape: B,H,S,W
    - interval shape: 1,n_G,1,n_V,1,n_H,1
    """

    def __init__(
        self,
        A_bit=8,
        B_bit=8,
        mode="raw",
        metric="L2_norm",
        search_round=1,
        eq_alpha=0.1,
        eq_beta=2,
        eq_n=100,
        parallel_eq_n=10,
        n_G_A=1,
        n_V_A=1,
        n_H_A=1,
        n_G_B=1,
        n_V_B=1,
        n_H_B=1,
        init_layerwise=False,
    ):
        super().__init__(A_bit=A_bit, B_bit=B_bit, mode=mode)
        self.metric = metric
        self.search_round = search_round
        self.eq_alpha = eq_alpha
        self.eq_beta = eq_beta
        self.eq_n = eq_n
        self.parallel_eq_n = parallel_eq_n
        self.n_G_A = n_G_A
        self.n_V_A = n_V_A
        self.n_H_A = n_H_A
        self.n_G_B = n_G_B
        self.n_V_B = n_V_B
        self.n_H_B = n_H_B
        # init these parameters in self.calibration_step1
        self.crb_groups_A = None
        self.crb_groups_B = None
        self.crb_rows_A = None
        self.crb_cols_A = None
        self.crb_rows_B = None
        self.crb_cols_B = None
        self.pad_groups_A = None
        self.pad_groups_B = None
        self.pad_rows_A = None
        self.pad_rows_B = None
        self.pad_cols_A = None
        self.pad_cols_B = None
        self.raw_grad = None
        self.init_layerwise = init_layerwise

    def quant_input_A(self, x):
        """quantize input activations"""
        x = F.pad(x, [0, self.pad_cols_A, 0, self.pad_rows_A, 0, self.pad_groups_A])
        x = x.view(
            -1,
            self.n_G_A,
            self.crb_groups_A,
            self.n_V_A,
            self.crb_rows_A,
            self.n_H_A,
            self.crb_cols_A,
        )
        x = (
            (x / self.A_interval)
            .round_()
            .clamp(-self.A_qmax, self.A_qmax - 1)
            .mul_(self.A_interval)
        )
        x = x.view(
            -1,
            self.n_G_A * self.crb_groups_A,
            self.n_V_A * self.crb_rows_A,
            self.n_H_A * self.crb_cols_A,
        )
        x = x[
            :,
            : x.shape[1] - self.pad_groups_A,
            : x.shape[2] - self.pad_rows_A,
            : x.shape[3] - self.pad_cols_A,
        ]
        return x

    def quant_input_B(self, x):
        """quantize inputs"""
        x = F.pad(x, [0, self.pad_cols_B, 0, self.pad_rows_B, 0, self.pad_groups_B])
        x = x.reshape(
            -1,
            self.n_G_B,
            self.crb_groups_B,
            self.n_V_B,
            self.crb_rows_B,
            self.n_H_B,
            self.crb_cols_B,
        )
        x = (
            (x / self.B_interval)
            .round_()
            .clamp(-self.B_qmax, self.B_qmax - 1)
            .mul_(self.B_interval)
        )
        x = x.view(
            -1,
            self.n_G_B * self.crb_groups_B,
            self.n_V_B * self.crb_rows_B,
            self.n_H_B * self.crb_cols_B,
        )
        x = x[
            :,
            : x.shape[1] - self.pad_groups_B,
            : x.shape[2] - self.pad_rows_B,
            : x.shape[3] - self.pad_cols_B,
        ]
        return x

    def quant_forward(self, A, B):
        """forward via quantized weights"""
        assert (
            self.calibrated is not None
        ), f"You should run calibrate_forward before run quant_forward for {self}"
        A_sim = self.quant_input_A(A)
        B_sim = self.quant_input_B(B)
        out = A_sim @ B_sim
        return out


class PTQSLBatchingQuantMatMul(PTQSLQuantMatMul):
    """post training quantization matrix multipliation"""

    def __init__(
        self,
        A_bit=8,
        B_bit=8,
        mode="raw",
        metric="L2_norm",
        search_round=1,
        eq_alpha=0.1,
        eq_beta=2,
        eq_n=100,
        parallel_eq_n=10,
        n_G_A=1,
        n_V_A=1,
        n_H_A=1,
        n_G_B=1,
        n_V_B=1,
        n_H_B=1,
        init_layerwise=False,
    ):
        super().__init__(
            A_bit=A_bit,
            B_bit=B_bit,
            mode=mode,
            metric=metric,
            search_round=search_round,
            eq_alpha=eq_alpha,
            eq_beta=eq_beta,
            eq_n=eq_n,
            parallel_eq_n=parallel_eq_n,
            n_G_A=n_G_A,
            n_V_A=n_V_A,
            n_H_A=n_H_A,
            n_G_B=n_G_B,
            n_V_B=n_V_B,
            n_H_B=n_H_B,
            init_layerwise=init_layerwise,
        )

        self.calib_size = None
        self.calib_batch_size = None
        self.calib_need_batching = None
        self.calibrated = False

    def _initialize_calib_parameters(self):
        """
        set parameters for feeding calibration data
        """
        self.calib_size = int(self.raw_input[0].shape[0])
        self.calib_batch_size = int(self.raw_input[0].shape[0])
        while True:
            numel = (
                (
                    self.raw_input[0].numel()
                    + self.raw_input[1].numel()
                    + 2 * self.raw_out.numel()
                )
                / self.calib_size
                * self.calib_batch_size
            )  # number of parameters on GPU
            self.parallel_eq_n = int((3 * 1024 * 1024 * 1024 / 4) // numel)
            if self.parallel_eq_n <= 1:
                self.calib_need_batching = True
                self.calib_batch_size //= 2
            else:
                break

    def _get_padding_parameters(self, A, B):
        """
        We adopt a head-wise quantization here
        """

        self.n_G_A = A.shape[1]
        self.n_G_B = B.shape[1]
        super()._get_padding_parameters(A, B)

    def _initialize_intervals(self):
        # pad A and B for future quantization
        self._get_padding_parameters(
            self.raw_input[0], self.raw_input[1]
        )  # put it here because hessian does not use calibration step 1

        # initialize intervals with minmax intervals
        tmp_A_intervals = []
        tmp_B_intervals = []
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            A, B = (
                self.raw_input[0][b_st:b_ed].cuda(),
                self.raw_input[1][b_st:b_ed].cuda(),
            )
            if self.init_layerwise:
                A_interval = (
                    (A.abs().max() / (self.A_qmax - 0.5))
                    .detach()
                    .view(1, 1, 1, 1, 1, 1, 1)
                    .repeat(1, self.n_G_A, 1, self.n_V_A, 1, self.n_H_A, 1)
                )
                B_interval = (
                    (B.abs().max() / (self.B_qmax - 0.5))
                    .detach()
                    .view(1, 1, 1, 1, 1, 1, 1)
                    .repeat(1, self.n_G_B, 1, self.n_V_B, 1, self.n_H_B, 1)
                )
            else:
                A_pad = (
                    F.pad(
                        A,
                        [0, self.pad_cols_A, 0, self.pad_rows_A, 0, self.pad_groups_A],
                    )
                    .unsqueeze(0)
                    .view(
                        1,
                        -1,
                        self.n_G_A,
                        self.crb_groups_A,
                        self.n_V_A,
                        self.crb_rows_A,
                        self.n_H_A,
                        self.crb_cols_A,
                    )
                )
                B_pad = (
                    F.pad(
                        B,
                        [0, self.pad_cols_B, 0, self.pad_rows_B, 0, self.pad_groups_B],
                    )
                    .unsqueeze(0)
                    .view(
                        1,
                        -1,
                        self.n_G_B,
                        self.crb_groups_B,
                        self.n_V_B,
                        self.crb_rows_B,
                        self.n_H_B,
                        self.crb_cols_B,
                    )
                )
                A_interval = (
                    (
                        A_pad.abs().amax([0, 1, 3, 5, 7], keepdim=True)
                        / (self.A_qmax - 0.5)
                    )
                    .detach()
                    .squeeze(0)
                )  # shape: 1,n_G,1,n_V,1,n_H,1
                B_interval = (
                    (
                        B_pad.abs().amax([0, 1, 3, 5, 7], keepdim=True)
                        / (self.B_qmax - 0.5)
                    )
                    .detach()
                    .squeeze(0)
                )  # shape: 1,n_G,1,n_V,1,n_H,1
            tmp_A_intervals.append(A_interval)
            tmp_B_intervals.append(B_interval)
        self.A_interval = torch.cat(tmp_A_intervals, dim=0).amax(0, keepdim=True)
        self.B_interval = torch.cat(tmp_B_intervals, dim=0).amax(0, keepdim=True)

    def _get_similarity(
        self, tensor_raw, tensor_sim, metric=None, dim=-1, raw_grad=None
    ):
        """
        tensor_raw: *, features, *
        tensor_sim: *, features, *
        similarity: *
        It's your job to calculate mean on non-feature * dims!

        Similarity without inherent feature structure is more welcome to parallelism.
        """
        if metric == "cosine":
            similarity = F.cosine_similarity(
                tensor_raw, tensor_sim, dim=dim
            )  # should only support dim=-1 and cannot be paralleled
        elif metric == "pearson":
            similarity = F.cosine_similarity(
                tensor_raw - torch.mean(tensor_raw, dim=dim, keepdim=True),
                tensor_sim - torch.mean(tensor_sim, dim=dim, keepdim=True),
                dim=dim,
            )  # should only support dim=-1 and cannot be paralleled
        else:
            if metric == "L1_norm":
                similarity = -torch.abs(tensor_raw - tensor_sim)
            elif metric == "L2_norm":
                similarity = -((tensor_raw - tensor_sim) ** 2)
            elif metric == "linear_weighted_L2_norm":
                similarity = -tensor_raw.abs() * (tensor_raw - tensor_sim) ** 2
            elif metric == "square_weighted_L2_norm":
                similarity = -((tensor_raw * (tensor_raw - tensor_sim)) ** 2)
            elif metric == "hessian":
                assert raw_grad is not None, "No raw_grad in PTQSLBatchingQuantMatMul!"
                raw_grad = raw_grad.reshape_as(tensor_raw)
                similarity = -((raw_grad * (tensor_raw - tensor_sim)) ** 2)
            else:
                raise NotImplementedError(f"metric {metric} not implemented!")
            similarity = torch.mean(similarity, dim=dim)
        return similarity

    def _search_best_A_interval(self, A_interval_candidates):
        """
        Modularization of searching best interval
        """
        tmp_A_interval = self.A_interval.unsqueeze(0)  # shape: 1,1,n_G,1,n_V,1,n_H,1
        # out-of-loop optimization
        for v, h in product(range(self.n_V_A), range(self.n_H_A)):
            batch_similarities = (
                []
            )  # similarities, need to concatenate and calculate sum
            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                A = self.raw_input[0][b_st:b_ed].cuda()
                A_pad = (
                    F.pad(
                        A,
                        [0, self.pad_cols_A, 0, self.pad_rows_A, 0, self.pad_groups_A],
                    )
                    .unsqueeze(0)
                    .view(
                        1,
                        -1,
                        self.n_G_A,
                        self.crb_groups_A,
                        self.n_V_A,
                        self.crb_rows_A,
                        self.n_H_A,
                        self.crb_cols_A,
                    )
                )
                B = self.raw_input[1][b_st:b_ed].cuda()
                B_sim = self.quant_input_B(B).unsqueeze(0)  # shape: 1,b,H,dim2,dim3
                raw_out = self.raw_out[b_st:b_ed].unsqueeze(0).cuda()
                raw_grad = self.raw_grad[b_st:b_ed].cuda()
                similarities = []
                for p_st in range(0, self.eq_n, self.parallel_eq_n):
                    p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                    # quantize A
                    cur_A_interval = tmp_A_interval.repeat(
                        p_ed - p_st, 1, 1, 1, 1, 1, 1, 1
                    )
                    cur_A_interval[
                        :, :, :, :, v : v + 1, :, h : h + 1, :
                    ] = A_interval_candidates[
                        p_st:p_ed, :, :, :, v : v + 1, :, h : h + 1, :
                    ]
                    A_sim = (
                        (A_pad / cur_A_interval)
                        .round_()
                        .clamp_(-self.A_qmax, self.A_qmax - 1)
                        .mul_(cur_A_interval)
                    )
                    A_sim = A_sim.view(
                        p_ed - p_st,
                        -1,
                        A.shape[1] + self.pad_groups_A,
                        A.shape[2] + self.pad_rows_A,
                        A.shape[3] + self.pad_cols_A,
                    )  # shape: parallel_eq_n,B,H*,dim1*,dim2* (* stand for padding)
                    A_sim = A_sim[
                        :, :, : A.shape[1], : A.shape[2], : A.shape[3]
                    ]  # shape: parallel_eq_n,b,H,dim1,dim2
                    # quantize B, this quantization is optimized out of loop
                    # calculate similarity and store them
                    out_sim = A_sim @ B_sim  # shape: parallel_eq_n,B,H,dim1,dim3
                    similarity = self._get_similarity(
                        raw_out, out_sim, self.metric, raw_grad=raw_grad
                    )  # shape: parallel_eq_n,b,H,dim1
                    similarity = similarity.mean(
                        [3]
                    )  # shape: parallel_eq_n,b,H (remaining mean operation will be done later on)
                    similarity = similarity.sum(
                        dim=1, keepdim=True
                    )  # shape: parallel_eq_n,1,H
                    similarities.append(similarity)
                # calculate best similarity for this block
                similarities = torch.cat(similarities, 0)  # shape: eq_n,1,H
                batch_similarities.append(similarities)
            batch_similarities = torch.cat(batch_similarities, dim=1).sum(
                dim=1, keepdim=False
            )  # shape: eq_n,H
            batch_similarities = (
                F.pad(batch_similarities, [0, self.pad_groups_A])
                .view(self.eq_n, self.n_G_A, self.crb_groups_A)
                .mean(-1)
            )  # shape: eq_n, n_G_A
            best_index = torch.argmax(batch_similarities, dim=0, keepdim=False).view(
                1, 1, -1, 1, 1, 1, 1, 1
            )
            tmp_A_interval[:, :, :, :, v : v + 1, :, h : h + 1, :] = torch.gather(
                A_interval_candidates[:, :, :, :, v : v + 1, :, h : h + 1, :],
                dim=0,
                index=best_index,
            )
        self.A_interval = tmp_A_interval.squeeze(0)

    def _search_best_B_interval(self, B_interval_candidates):
        """
        Modularization of searching best interval
        """
        tmp_B_interval = self.B_interval.unsqueeze(0)  # shape: 1,1,n_G,1,n_V,1,n_H,1
        # out-of-loop optimization
        for v, h in product(range(self.n_V_B), range(self.n_H_B)):
            batch_similarities = (
                []
            )  # similarities, need to concatenate and calculate sum
            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                A = self.raw_input[0][b_st:b_ed].cuda()
                A_sim = self.quant_input_A(A).unsqueeze(0)  # shape: 1,B,H,dim1,dim2
                B = self.raw_input[1][b_st:b_ed].cuda()
                B_pad = (
                    F.pad(
                        B,
                        [0, self.pad_cols_B, 0, self.pad_rows_B, 0, self.pad_groups_B],
                    )
                    .unsqueeze(0)
                    .view(
                        1,
                        -1,
                        self.n_G_B,
                        self.crb_groups_B,
                        self.n_V_B,
                        self.crb_rows_B,
                        self.n_H_B,
                        self.crb_cols_B,
                    )
                )
                raw_out = self.raw_out[b_st:b_ed].unsqueeze(0).cuda()
                raw_grad = self.raw_grad[b_st:b_ed].cuda()
                similarities = []
                for p_st in range(0, self.eq_n, self.parallel_eq_n):
                    p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                    # quantize A, this quantization is optimized out of loop
                    # quantize B
                    cur_B_interval = tmp_B_interval.repeat(
                        p_ed - p_st, 1, 1, 1, 1, 1, 1, 1
                    )
                    cur_B_interval[
                        :, :, :, :, v : v + 1, :, h : h + 1, :
                    ] = B_interval_candidates[
                        p_st:p_ed, :, :, :, v : v + 1, :, h : h + 1, :
                    ]
                    B_sim = (
                        (B_pad / cur_B_interval)
                        .round_()
                        .clamp_(-self.B_qmax, self.B_qmax - 1)
                        .mul_(cur_B_interval)
                    )
                    B_sim = B_sim.view(
                        p_ed - p_st,
                        -1,
                        B.shape[1] + self.pad_groups_B,
                        B.shape[2] + self.pad_rows_B,
                        B.shape[3] + self.pad_cols_B,
                    )  # shape: parallel_eq_n,b,H*,dim2*,dim3* (* stand for padding)
                    B_sim = B_sim[
                        :, :, : B.shape[1], : B.shape[2], : B.shape[3]
                    ]  # shape: parallel_eq_n,b,H,dim2,dim3
                    # calculate similarity and store them
                    out_sim = A_sim @ B_sim  # shape: parallel_eq_n,b,H,dim1,dim3
                    similarity = self._get_similarity(
                        raw_out, out_sim, self.metric, raw_grad=raw_grad
                    )  # shape: parallel_eq_n,b,H,dim1
                    similarity = similarity.mean(
                        [3]
                    )  # shape: parallel_eq_n,b,H (remaining mean operation will be done later on)
                    similarity = similarity.sum(
                        dim=1, keepdim=True
                    )  # shape: parallel_eq_n,1,H
                    similarities.append(similarity)
                # calculate best similarity for this block
                similarities = torch.cat(similarities, 0)  # shape: eq_n,1,H
                batch_similarities.append(similarities)
            batch_similarities = torch.cat(batch_similarities, dim=1).sum(
                dim=1, keepdim=False
            )  # shape: eq_n,H
            batch_similarities = (
                F.pad(batch_similarities, [0, self.pad_groups_B])
                .view(self.eq_n, self.n_G_B, self.crb_groups_B)
                .mean(-1)
            )  # shape: eq_n, n_G_B
            best_index = torch.argmax(batch_similarities, dim=0, keepdim=False).view(
                1, 1, -1, 1, 1, 1, 1, 1
            )
            tmp_B_interval[:, :, :, :, v : v + 1, :, h : h + 1, :] = torch.gather(
                B_interval_candidates[:, :, :, :, v : v + 1, :, h : h + 1, :],
                dim=0,
                index=best_index,
            )
        self.B_interval = tmp_B_interval.squeeze(0)

    def calibration_step2(self):
        """second calibration step"""
        self._initialize_calib_parameters()
        self._initialize_intervals()
        A_interval_candidates = torch.tensor(
            [
                self.eq_alpha + i * (self.eq_beta - self.eq_alpha) / self.eq_n
                for i in range(self.eq_n + 1)
            ]
        ).cuda().view(-1, 1, 1, 1, 1, 1, 1, 1) * self.A_interval.unsqueeze(0)
        B_interval_candidates = torch.tensor(
            [
                self.eq_alpha + i * (self.eq_beta - self.eq_alpha) / self.eq_n
                for i in range(self.eq_n + 1)
            ]
        ).cuda().view(-1, 1, 1, 1, 1, 1, 1, 1) * self.B_interval.unsqueeze(0)
        for _ in range(self.search_round):
            # search for best A interval
            self._search_best_A_interval(A_interval_candidates)
            # search for best B interval
            self._search_best_B_interval(B_interval_candidates)
        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad


class SoSPTQSLBatchingQuantMatMul(PTQSLBatchingQuantMatMul):
    """batching for memory efficient inference"""

    def __init__(
        self,
        A_bit=8,
        B_bit=8,
        mode="raw",
        metric="L2_norm",
        search_round=1,
        eq_alpha=0.1,
        eq_beta=2,
        eq_n=100,
        parallel_eq_n=10,
        n_G_A=1,
        n_V_A=1,
        n_H_A=1,
        n_G_B=1,
        n_V_B=1,
        n_H_B=1,
        init_layerwise=False,
        split=None,
    ):
        super().__init__(
            A_bit=A_bit,
            B_bit=B_bit,
            mode=mode,
            metric=metric,
            search_round=search_round,
            eq_alpha=eq_alpha,
            eq_beta=eq_beta,
            eq_n=eq_n,
            parallel_eq_n=parallel_eq_n,
            n_G_A=n_G_A,
            n_V_A=n_V_A,
            n_H_A=n_H_A,
            n_G_B=n_G_B,
            n_V_B=n_V_B,
            n_H_B=n_H_B,
            init_layerwise=init_layerwise,
        )
        self.n_G_A = 1
        self.n_V_A = 1
        self.n_H_A = 1
        # with proper hardware implementation, we don't need to use a sign bit anymore
        self.A_qmax = 2 ** (self.A_bit - 1)
        self.split = split
        if split is not None:
            self.A_interval = self.split / (self.A_qmax - 1)

    def quant_input_A(self, x):
        """quant input"""
        x_high = (x.clamp(self.split, 1) * (self.A_qmax - 1)).round_().clamp_(
            0, self.A_qmax - 1
        ) / (self.A_qmax - 1)
        x_low = (x.clamp(0, self.split) / self.A_interval).round_().clamp_(
            0, self.A_qmax - 1
        ) * self.A_interval
        return x_high + x_low

    def _search_best_A_interval(self, split_candidates):
        """search for best interval"""
        batch_similarities = []
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            A = self.raw_input[0][b_st:b_ed].unsqueeze(0).cuda()
            B = self.raw_input[1][b_st:b_ed].unsqueeze(0).cuda()
            B_sim = B
            raw_out = self.raw_out[b_st:b_ed].unsqueeze(0).cuda()
            raw_grad = self.raw_grad[b_st:b_ed].cuda()
            similarities = []
            for i in range(len(split_candidates)):
                # quantize A
                cur_A_interval = split_candidates[i] / (self.A_qmax - 1)
                A_high = (
                    A.clamp(split_candidates[i], 1) * (self.A_qmax - 1)
                ).round_().clamp_(0, self.A_qmax - 1) / (self.A_qmax - 1)
                A_low = (
                    A.clamp(0, split_candidates[i]) / cur_A_interval
                ).round_().clamp_(0, self.A_qmax - 1) * cur_A_interval
                A_sim = A_high + A_low  # shape: 1,b,H,S,S
                # quantize B, this quantization is optimized out of loop
                # calculate similarity and store them (dim1=dim2=S, dim3=W)
                out_sim = A_sim @ B_sim  # shape: 1,b,H,dim1,dim3
                similarity = self._get_similarity(
                    raw_out, out_sim, self.metric, raw_grad=raw_grad
                )  # shape: parallel_eq_n,b,H,dim1
                similarity = similarity.mean([2, 3])  # shape: parallel_eq_n, b
                similarity = similarity.sum(dim=1, keepdim=True)  # parallel_eq_n, 1
                similarities.append(similarity)
            # calculate best similarity for this block
            similarities = torch.cat(similarities, 0)  # shape: eq_n, 1
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=1).sum(
            dim=1, keepdim=False
        )  # shape: eq_n
        best_index = torch.argmax(batch_similarities, dim=0, keepdim=False)
        self.split = split_candidates[best_index]
        self.A_interval = self.split / (self.A_qmax - 1)
        # debugging
        # print(f"best split: {self.split}")

    def calibration_step2(self):
        """run calibaration"""
        self._initialize_calib_parameters()
        self._initialize_intervals()
        A_split_candidates = torch.tensor([2 ** (-i) for i in range(20)]).cuda()
        B_interval_candidates = torch.tensor(
            [
                self.eq_alpha + i * (self.eq_beta - self.eq_alpha) / self.eq_n
                for i in range(self.eq_n + 1)
            ]
        ).cuda().view(-1, 1, 1, 1, 1, 1, 1, 1) * self.B_interval.unsqueeze(0)
        for _ in range(self.search_round):
            # search for best A interval
            self._search_best_A_interval(A_split_candidates)
            # search for best B interval
            self._search_best_B_interval(B_interval_candidates)
        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad
