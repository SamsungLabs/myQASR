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
Unless required by applicable law or agreed to in writing, software distributed under 
the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import jiwer
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from models.Qwav2vec import QWav2Vec
from models.Wav2VecConformer import Wav2VecConformer


def get_model(args):
    """Returns the loaded model from huggingface with pretrained weights"""
    # Initialize variables for layer counts
    conv_layers = att_layers = 0

    # Handle model loading and setup based on model size
    if args.model in ["wav2vec-base", "wav2vec-large"]:
        model_size = "base" if args.model == "wav2vec-base" else "large"
        model = QWav2Vec(args, model_size=model_size, method="brute")
        model_type = "small" if model_size == "base" else "large"
        print(f"[+] Using {model_type} wav2vec2 model with libri dataset.")
        conv_layers = len(model.feature_extractor)
        att_layers = len(model.att) if hasattr(model, "att") else 0
    elif args.model == "wav2vec-conformer":
        model = Wav2VecConformer(args)
    else:
        print(f"[!] Error {args.model} not implemented")
        raise NotImplementedError

    # Print total layer composition if applicable
    if args.model in ["wav2vec-base", "wav2vec-large"]:
        print("[+] Total layer composition")
        if args.conv and args.linear:
            layers = model.feature_extractor + model.att
            print(f"     [-] Conv layers: {conv_layers}")
            print(f"     [-] Att layers: {att_layers}")
        elif args.conv and not args.linear:
            layers = model.feature_extractor
            print(f"     [-] Conv layers: {conv_layers}")
        elif args.linear and not args.conv:
            layers = model.att
            print(f"     [-] Att layers: {att_layers}")
        else:
            layers = []
    elif args.model == "wav2vec-conformer":
        layers = model.create_layers()

    return model, layers


def test_speech_commands(net, data):
    """test the performance of speech commands dataset"""
    transcriptions = []
    labels = []
    for input_values, label in data:
        logits = net(input_values.cuda()).logits
        predicted_id = torch.argmax(logits, dim=-1)  # pylint: disable=no-member
        transcriptions.extend(predicted_id.cpu())
        labels.extend(label)
    acc = accuracy_score(labels, transcriptions)
    return acc, 0


def test_libri(net, test_loader, processor):
    """test performance of librispeech dataset"""
    with torch.no_grad():
        hypotheses = []
        references = []

        for mels, texts in tqdm(test_loader):
            results = net(mels.cuda()).logits
            ids = torch.argmax(results, dim=-1)  # pylint: disable=no-member
            transcription = processor.batch_decode(ids)
            hypotheses.extend(transcription)
            references.append(texts)

        wer = jiwer.wer(references, hypotheses)
        cer = jiwer.cer(references, hypotheses)
        return wer, cer
