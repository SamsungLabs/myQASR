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
from sklearn.metrics import accuracy_score
from transformers import AutoModelForAudioClassification, Wav2Vec2Processor


def compute_all(ground_truth, hypothesis):
    """compute metrics"""
    measures = jiwer.compute_measures(ground_truth, hypothesis)
    cer = jiwer.cer(ground_truth, hypothesis)
    return cer, measures


class Wav2VecConformer:
    """Initialse a quantized WAV2VEC2 Model for evaluation or training."""

    def __init__(self, config, model_size="base", method="default"):
        config = vars(config)

        if config["data"] == "speech_commands":
            if config["test_lang"] or config["tune_lang"]:
                self.processor = Wav2Vec2Processor.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                )
            else:
                self.processor = Wav2Vec2Processor.from_pretrained(
                    f"facebook/wav2vec2-{model_size}-960h"
                )
            self.basemodel = AutoModelForAudioClassification.from_pretrained(
                "juliensimon/wav2vec2-conformer-rel-pos-large-finetuned-speech-commands"
            ).cuda()
        else:
            raise NotImplementedError(
                "Currently WAV2VECCONFORMER only supports Speech Commands Data Set"
            )

        self.freq_boost = 0
        # for i in [1, 2, 3, 4, 9]:
        # update for base vs large - how many layers?
        # self.output = ["lm_head"]
        self.config = config

    def create_layers(self):
        """create layers for quantizing"""
        layers = []
        for name, param in self.basemodel.named_parameters():
            # print(name, (param.nelement() / 4) / 1024**2)
            if (
                "weight" in name
                and ("norm" not in name)
                and ("embed_positions" not in name)
                and ("embed_tokens" not in name)
                and ("pos_conv_embed" not in name)
            ):
                layers.append(name)
            #     print(f"{name}: Y")
            # else:
            #     print(f"{name}: N")
        return layers

    def tidy_text(self, batch):
        """clean text for evaluation"""
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
        """map embeds to predictions"""
        input_values = self.processor(
            batch["audio"][0]["array"], return_tensors="pt", sampling_rate=16000
        ).input_values

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
        """evaluate model"""
        # result = data.map(self._map_to_pred, batched=True,
        #                   batch_size=1, remove_columns=["audio"])
        transcriptions = []
        texts = []
        for input_values, text in data:
            logits = self.basemodel(input_values.cuda()).logits
            predicted_id = torch.argmax(logits, dim=-1)  # pylint: disable=no-member
            # transcription = self.processor.batch_decode(predicted_ids)
            transcriptions.extend(predicted_id.cpu())
            texts.extend(text)
        acc = accuracy_score(texts, transcriptions)
        return acc, {"cer": 0, "wer": acc, "mil": 0, "mer": 0, "wil": 0}

    def get_size(self):
        """calculate size of model"""
        param_size = 0.0
        for param in self.basemodel.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0.0
        for buffer in self.basemodel.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size = param_size + buffer_size
        size = size / 1024**2
        return size
