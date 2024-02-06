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

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import jiwer
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def compute_all(ground_truth, hypothesis):
    """compute all metrics"""
    measures = jiwer.compute_measures(ground_truth, hypothesis)
    cer = jiwer.cer(ground_truth, hypothesis)
    return cer, measures


class QWav2Vec:
    """Initialse a quantized WAV2VEC2 Model for evaluation or training."""

    def __init__(self, config, model_size="base", method="default"):
        config = vars(config)

        if config["data"] == "libri" or config["data"] == "speech_commands":
            if config["test_lang"] or config["tune_lang"]:
                self.processor = Wav2Vec2Processor.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                )
            else:
                self.processor = Wav2Vec2Processor.from_pretrained(
                    f"facebook/wav2vec2-{model_size}-960h"
                )
            if model_size == "base":
                print("[logging] Using Base model Wav2Vec")
                self.basemodel = Wav2Vec2ForCTC.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                ).to("cuda")
            elif model_size == "large":
                print("[logging] Using Large model Wav2Vec")
                self.basemodel = Wav2Vec2ForCTC.from_pretrained(
                    "facebook/wav2vec2-large-960h"
                ).to("cuda")
            else:
                print("! Select a model size --large or --base")
                raise NotImplementedError

        elif config["model_ds"] == "cv-en":
            self.basemodel = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
            if config["test_lang"] or config["tune_lang"]:
                self.processor = Wav2Vec2Processor.from_pretrained(
                    "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
                )
            else:
                self.processor = Wav2Vec2Processor.from_pretrained(
                    f"facebook/wav2vec2-{model_size}-960h"
                )

        else:
            raise NotImplementedError(
                "Currently WAV2VEC only supports pretrained libri"
            )

        if model_size == "base":
            conv_range = 7
            att_range = 12
        else:
            conv_range = 7
            att_range = 24

        if config["data"] == "common_voice":
            self.text = "sentence"
        else:
            self.text = "text"

        self.feature_extractor = [
            f"wav2vec2.feature_extractor.conv_layers.{i}.conv.weight"
            for i in range(conv_range)
        ]
        self.encoder_blocks = []
        self.encoder = []
        self.att = []
        self.freq_boost = 0
        # for i in [1, 2, 3, 4, 9]:
        # update for base vs large - how many layers?
        for i in range(att_range):
            if method == "default":
                self.att = []
            for a in ["q", "k", "v", "out"]:
                self.att += [f"wav2vec2.encoder.layers.{i}.attention.{a}_proj.weight"]
            # self.att += [f"wav2vec2.encoder.layers.{i}.intermediate_dense"]
            # self.att += [f"wav2vec2.encoder.layers.{i}.output_dense"]
            self.att += [
                f"wav2vec2.encoder.layers.{i}.feed_forward.output_dense.weight"
            ]
            self.att += [
                f"wav2vec2.encoder.layers.{i}.feed_forward.intermediate_dense.weight"
            ]
            # self.att += [
            #     f"wav2vec2.encoder.layers.{i}.final_layer_norm"]

            # self.att += [
            #     f"wav2vec2.encoder.layers.{i}.layer_norm"]
            self.encoder_blocks.append(self.att)
        # self.encoder_blocks = [f"wav2vec2.encoder.layers.{i}" for i in range(12)]
        self.output = ["lm_head"]
        self.config = config

    def tidy_text(self, batch):
        """sanitize text output from model"""
        chars_to_ignore_regex = r'[\,\?\.\!\-\;\:"]'  # pylint: disable=W1401
        # speech, _ = torchaudio.load(batch["path"])
        # batch["speech"] = resampler.forward(speech.squeeze(0)).numpy()
        # batch["sampling_rate"] = resampler.new_freq
        # batch = re.sub(
        #     chars_to_ignore_regex, '', batch).lower().replace("’", "'")
        # # print(batch)

        batch = re.sub(chars_to_ignore_regex, "", batch).lower().replace("’", "'")
        # batch = batch[1:]
        # batch["transcription"] = re.sub(
        #     r'\b(\w+)( \1\b)+', r'\1', batch["transcription"])

        # batch["transcription"] = re.sub(
        #     r'((\b\w+\b.{1,2}\w+\b)+).+\1', r'\1', batch["transcription"], flags=re.I)

        # batch[self.text] = re.sub(
        #     chars_to_ignore_regex, '', batch[self.text]).lower().replace("’", "'")
        # batch[self.text] = batch[self.text][1:]
        # batch[self.text] = re.sub(
        #     r'\b(\w+)( \1\b)+', r'\1', batch[self.text])

        # batch[self.text] = re.sub(
        #     r'((\b\w+\b.{1,2}\w+\b)+).+\1', r'\1', batch[self.text], flags=re.I)

        return batch

    def _map_to_pred(self, batch):
        """map the outputs to predictions"""
        input_values = self.processor(
            batch["audio"][0]["array"], return_tensors="pt", sampling_rate=16000
        ).input_values

        # functions for compression based on spectral properties
        # print(spectral_properties(input_values[0].squeeze(0),16000)["mean"])
        # input_values = torchaudio.functional.pitch_shift(input_values, 16000, 4)
        # print(spectral_properties(input_values[0].squeeze(0),16000)["mean"])
        # exit()
        # if spectral_properties(input_values[0].squeeze(0), 1600)["mean"] < 1000:
        #     input_values = torchaudio.functional.highpass_biquad(
        #         input_values, 16000, 1300)
        # shift frequency using naive pitch method -
        # better solution maybe dynamic compression of range of frequencies.
        # self.plt_spectograms(input_values, batch, alt="before")
        # input_values = input_values.to("cuda")

        with torch.no_grad():
            logits = self.basemodel(
                input_values.to(f"cuda:{self.config['gpu']}")
            ).logits

        predicted_ids = torch.argmax(logits, dim=-1)  # pylint: disable=no-member
        transcription = self.processor.batch_decode(predicted_ids)
        # get transcription - sentence for common and transcription for libri
        batch["transcription"] = transcription
        return batch

    def evaluate(self, data):
        """evaluate the model"""
        # result = data.map(self._map_to_pred, batched=True,
        #                   batch_size=1, remove_columns=["audio"])
        transcriptions = []
        texts = []
        with torch.no_grad():
            for input_values, text in data:
                logits = self.basemodel(input_values.cuda()).logits
                predicted_ids = torch.argmax(  # pylint: disable=no-member
                    logits, dim=-1
                )
                transcription = self.processor.batch_decode(predicted_ids)
                transcriptions.extend(transcription)
                texts.append(text)
        return compute_all(texts, transcriptions)

    def get_size(self):
        """get the size of the model"""
        param_size = 0.0
        for param in self.basemodel.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0.0
        for buffer in self.basemodel.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size = param_size + buffer_size
        size = size / 1024**2
        return size
