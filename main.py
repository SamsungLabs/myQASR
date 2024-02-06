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

# Libraries
import os
import argparse
import random
from importlib import reload, import_module
import numpy as np
import torch

# Local imports
from models.model_utils import get_model, test_speech_commands, test_libri
from utils import net_wrap
from utils.logging_utils import export_results

# from utils.models import get_net
from utils.dataloaders.data_utils import get_data
from utils.quant_calibration import CustomQuantCalibrator


def set_tune_test(input_args):
    """Deal with different tune-test configurations."""
    if input_args.tune_lang:
        tune = input_args.tune_lang
        test = input_args.test_lang
    elif input_args.tune_id:
        tune = input_args.tune_id
        test = input_args.test_id
    elif input_args.tune_gender:
        tune = input_args.tune_gender
        test = input_args.test_gender
    else:
        # Set False for no data partitions
        tune = 0
        test = 0
    return tune, test


def init_config(config_name):
    """initialize the config. Use reload to make sure it's fresh one!"""
    quant_cfg = import_module(f"configs.config")
    reload(quant_cfg)
    return quant_cfg


def set_uniform(cfg, w_bit, a_bit):
    """set uniform bit depths for weights and activations"""
    cfg.w_bit = {name: w_bit for name in cfg.conv_fc_name_list}
    cfg.a_bit = {name: a_bit for name in cfg.conv_fc_name_list}
    cfg.A_bit = {name: a_bit for name in cfg.matmul_name_list}
    cfg.B_bit = {name: a_bit for name in cfg.matmul_name_list}
    return cfg


def check_valid_args(args_valid):
    """Check if argurments are valid"""
    if args_valid.tune_lang or args_valid.test_lang:
        raise NotImplementedError("Langugages not supported currently")
    if args_valid.model in ["wav2vec-base", "wav2vec-small", "wav2vec-large"]:
        if args_valid.data != "libri":
            raise NotImplementedError(
                "Wav2Vec models only support '--data libri' at this time"
            )
    if args_valid.model in ["whisper-small", "whisper-large", "whisper-base"]:
        raise NotImplementedError("Whisper functionality disabled currently")
    if args_valid.model in ["wav2vec-conformer"]:
        if args_valid.data != "speech_commands":
            raise NotImplementedError(
                "Only speech_commands is currently compatible with wav2vec_conformer"
            )
        if args_valid.target < 500:
            print(
                "Reminder: Wav2Vec Conformer is over 2gb so \
                    compression to under 500mb may provide poor results"
            )


# def main(input_args):
#     """Main function."""
#     check_valid_args(input_args)

#     model_parent, _ = get_model(input_args)
#     model = model_parent.basemodel
#     tune_data, test_data = get_data(input_args)

#     quant_cfg = init_config(input_args.model)
#     wrapped_modules, _ = net_wrap.wrap_modules_in_net(model, quant_cfg)

#     quant_calibrator = CustomQuantCalibrator(
#         model, wrapped_modules, tune_data, sequential=input_args.sequential, batch_size=1
#     )
#     (
#         avg_bits,
#         min_bits,
#         max_bits,
#         std_bits,
#         compressed_size,
#         _,
#         global_model_dict,
#         sense_time,
#         calibration_time,
#         bits,
#     ) = quant_calibrator.batching_quant_calib(input_args)
#     compressed_size = get_model_size(global_model_dict)

#     export_dict = {
#         "model": input_args.model,
#         "data": input_args.data,
#         "sensitivity_method": input_args.sensitivity_method,
#         "calibration": input_args.calibration_method,
#         "sorting_method": input_args.sorting_method,
#         "avg_bits": round(avg_bits, 3),
#         "min_bits": min_bits,
#         "max_bits": max_bits,
#         "std_bits": std_bits,
#         "act_bit": input_args.a_bit,
#         "tune_samples": input_args.tune_samples,
#         "sample_len": input_args.tune_sample_len,
#         "compressed_size": round(compressed_size, 3),
#         "uniform": input_args.uniform,
#         "sequential": input_args.sequential,
#         "sensitivity_time": round(sense_time, 4),
#         "calibration_time": round(calibration_time, 4),
#         "percentile": input_args.percentile,
#         "bits": bits,
#     }

#     processor = model_parent.processor if input_args.model not in WHISPER_MODELS else None

#     def handle_data(data_type):
#         nonlocal processor
#         if data_type == "fleur":
#             raise NotImplementedError("Fleurs not currently supported")
#         if data_type == "libri":
#             wer, cer = test_libri(model, test_data, processor)
#             export_dict["wer"] = round(wer * 100, 3)
#             export_dict["cer"] = round(cer * 100, 3)
#         elif data_type == "speech_commands":
#             wer, cer = test_speech_commands(model, test_data)
#             export_dict["wer"] = round(wer * 100, 3)
#             export_dict["cer"] = round(cer * 100, 3)
#         else:
#             raise argparse.ArgumentError(
#                 None, "Dataset not identified. Should be 'libri' or 'speech_commands'"
#             )

#     handle_data(input_args.data)

#     tune, test = set_tune_test(input_args)
#     export_dict["tune_data"] = tune
#     export_dict["test_data"] = test

#     path = f"results/{input_args.experiment}/{input_args.model}/{input_args.data}/{test}"
#     os.makedirs(path, exist_ok=True)
#     export_dict["csv_file"] = (f"{path}/{tune}_{test}.csv",)

#     print(f"WER/ACC:{export_dict['wer']}:CER/OTHER:{export_dict['cer']}")
#     print(f"Results logged to {path}")
#     export_results(path, **export_dict)


WHISPER_MODELS = ("whisper-large", "whisper-base", "whisper-small")


def calibrate_model(model, tune_data, input_args):
    """Run calibration step"""
    quant_cfg = init_config(input_args.model)
    wrapped_modules, _ = net_wrap.wrap_modules_in_net(model, quant_cfg)

    quant_calibrator = CustomQuantCalibrator(
        model,
        wrapped_modules,
        tune_data,
        sequential=input_args.sequential,
        batch_size=1,
    )
    calibration_results = quant_calibrator.batching_quant_calib(input_args)

    (
        avg_bits,
        min_bits,
        max_bits,
        std_bits,
        compressed_size,
        _,
        _,
        sense_time,
        calibration_time,
        bits,
    ) = calibration_results
    return (
        avg_bits,
        min_bits,
        max_bits,
        std_bits,
        compressed_size,
        sense_time,
        calibration_time,
        bits,
    )


def test_quant(model, test_data, processor=None):
    """Return results on test data"""
    if processor is None:
        wer, cer = test_speech_commands(model, test_data)
    else:
        wer, cer = test_libri(model, test_data, processor)
    return wer, cer


def create_results_path(input_args, test):
    """location of saved results in csv format"""
    return (
        f"results/{input_args.experiment}/{input_args.model}/{input_args.data}/{test}"
    )


def main(input_args):
    """main function with input args from parser"""
    check_valid_args(input_args)

    model_parent, _ = get_model(input_args)
    model = model_parent.basemodel
    tune_data, test_data = get_data(input_args)

    processor = (
        model_parent.processor if input_args.model not in WHISPER_MODELS else None
    )

    (
        avg_bits,
        min_bits,
        max_bits,
        std_bits,
        compressed_size,
        sense_time,
        calibration_time,
        bits,
    ) = calibrate_model(model, tune_data, input_args)

    export_dict = {
        "model": input_args.model,
        "data": input_args.data,
        "sensitivity_method": input_args.sensitivity_method,
        "calibration": input_args.calibration_method,
        "sorting_method": input_args.sorting_method,
        "avg_bits": round(avg_bits, 3),
        "min_bits": min_bits,
        "max_bits": max_bits,
        "std_bits": std_bits,
        "act_bit": input_args.a_bit,
        "tune_samples": input_args.tune_samples,
        "sample_len": input_args.tune_sample_len,
        "compressed_size": round(compressed_size, 3),
        "uniform": input_args.uniform,
        "sequential": input_args.sequential,
        "sensitivity_time": round(sense_time, 4),
        "calibration_time": round(calibration_time, 4),
        "percentile": input_args.percentile,
        "bits": bits,
    }

    wer, cer = test_quant(model, test_data, processor)
    export_dict["wer"] = round(wer * 100, 3)
    export_dict["cer"] = round(cer * 100, 3)

    tune, test = set_tune_test(input_args)
    export_dict["tune_data"] = tune
    export_dict["test_data"] = test

    path = create_results_path(input_args, test)
    os.makedirs(path, exist_ok=True)
    export_dict["csv_file"] = (f"{path}/{tune}_{test}.csv",)

    print(f"WER/ACC:{export_dict['wer']}:CER/OTHER:{export_dict['cer']}")
    print(f"Results logged to {path}")
    export_results(path, **export_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MyQASR - Label and Data Free quantization"
    )
    parser.add_argument(
        "--model", type=str, default="wav2vec-base", help="Base or large wav2vec model"
    )

    parser.add_argument("--data", type=str, default="libri", help="Libri only atm")

    parser.add_argument("--tune_gender", type=str, default=None, help="gender to tune")

    parser.add_argument("--test_gender", type=str, default=None, help="gender to tune")

    parser.add_argument("--test_lang", type=str, default=None, help="lang to tune")

    parser.add_argument("--tune_lang", type=str, default=None, help="language to tune")

    parser.add_argument(
        "--tune_id", type=int, default=None, help="id to tune (librispeech specific)"
    )

    parser.add_argument(
        "--test_id",
        type=int,
        nargs="+",
        default=None,
        help="id to test (librispeech specific)",
    )

    parser.add_argument(
        "--tune_sample_len", type=int, default=25000, help="len of sample to tune"
    )

    parser.add_argument(
        "--target", type=float, default=None, help="Target for quantization"
    )

    parser.add_argument(
        "--conv", default=False, help="Include Conv1d layers", action="store_true"
    )

    parser.add_argument(
        "--linear", default=False, help="Include Linear layers", action="store_true"
    )
    parser.add_argument("--uniform", default=False, action="store_true")

    parser.add_argument(
        "--results_path",
        type=str,
        default="results.csv",
        help="specify results path for csv file",
    )

    parser.add_argument(
        "--tune_samples", type=int, default=32, help="number of tuning samples"
    )

    parser.add_argument(
        "--w_bit", type=int, default=8, help="Weight bit for uniform or min max"
    )

    parser.add_argument(
        "--a_bit", type=int, default=8, help="activation bit for uniform or min max"
    )

    parser.add_argument("--min_bit", type=int, default=0, help="min bit for min_max")

    parser.add_argument("--max_bit", type=int, default=0, help="max bit for min max")

    parser.add_argument("--bs", type=int, default=1, help="batch size")

    parser.add_argument("--seed", type=int, default=42, help="random seed")

    parser.add_argument(
        "--percentile",
        type=float,
        default=0.9,
        help="percentile for sensitivity methods",
    )

    parser.add_argument(
        "--sensitivity_method",
        type=str,
        default="median",
        help="method for selecting order of quant (median, max, abs_max, l1, l2, kl, spec) ",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="testing",
        help="experiment name for path extension",
    )

    parser.add_argument(
        "--sorting_method",
        type=str,
        default="uniform_reverse",
        help="method for ordering the bits (min_max, uniform, uniform_reverse, shuffle)",
    )

    parser.add_argument(
        "--calibration_method",
        type=str,
        default="cosine",
        help="method for calibrating activations and weights \
            (cosine, hessian, L1_norm, L2_norm, linear_weighted_L2_norm, linear_weighted_L1_norm",
    )

    parser.add_argument(
        "--sequential",
        default=False,
        action="store_true",
        help="Quantize layer by layer with quantized previous layer",
    )

    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="a default test environment for debugging",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)
