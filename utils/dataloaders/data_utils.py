"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Edward Fish (edward.fish@samsung.com; edward.fish@surrey.ac.uk)
Umberto Michieli (u.michieli@samsung.com)
Mete Ozay (m.ozay@samsung.com)

Licensed under the Creative Commons 
Attribution-NonCommercial-ShareAlike 4.0 International 
(CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at 
https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, 
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import io
import os
import re
import tarfile
import urllib

import pandas as pd
import torch
import torchaudio
import torchvision
import whisper
from datasets import Audio, load_dataset
from scipy.io import wavfile
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm
from transformers import Wav2Vec2Processor

csv_p = "utils/dataloaders/SPEAKERS.csv"


class Fleurs(Dataset):
    """
    A simple class to wrap Fleurs and subsample
    a portion of the dataset as needed.
    """

    def __init__(
        self,
        lang,
        split="test",
        subsample_rate=5,
        device=0,
        sample_len=10000,
        tune=False,
    ):
        url = f"https://storage.googleapis.com/xtreme_translations/FLEURS102/{lang}.tar.gz"
        tar_path = os.path.expanduser(f"~/.cache/fleurs/{lang}.tgz")
        os.makedirs(os.path.dirname(tar_path), exist_ok=True)
        self.tune = tune

        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

        if not os.path.exists(tar_path):
            with urllib.request.urlopen(url) as source, open(tar_path, "wb") as output:
                with tqdm(
                    total=int(source.info().get("Content-Length")),
                    ncols=80,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as loop:
                    while True:
                        buffer = source.read(8192)
                        if not buffer:
                            break

                        output.write(buffer)
                        loop.update(len(buffer))

        labels = {}
        all_audio = {}
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                name = member.name
                if name.endswith(f"{split}.tsv"):
                    labels = pd.read_table(
                        tar.extractfile(member),
                        names=(
                            "id",
                            "file_name",
                            "raw_transcription",
                            "transcription",
                            "_",
                            "num_samples",
                            "gender",
                        ),
                    )

                if f"/{split}/" in name and name.endswith(".wav"):
                    audio_bytes = tar.extractfile(member).read()
                    all_audio[os.path.basename(name)] = wavfile.read(
                        io.BytesIO(audio_bytes)
                    )[1]

        self.labels = labels.to_dict("records")[::subsample_rate]
        self.all_audio = all_audio
        self.device = device
        self.max_len = sample_len

    def __len__(self):
        """gets length of labels"""
        return len(self.labels)

    def __getitem__(self, item):
        """return item from datset"""
        record = self.labels[item]
        audio = torch.from_numpy(self.all_audio[record["file_name"]].copy())
        if self.tune:
            audio = audio[:100000]
        audio = whisper.pad_or_trim(audio)
        audio = whisper.log_mel_spectrogram(audio)
        text = record["transcription"]
        return (audio, text)

    # def custom_collate(self):
    #     audio_ls = []
    #     label_ls = []

    #     for d in data:
    #         inputs, labels = d
    #         audio_ls.append(inputs)
    #         labels_ls.append(labels)
    #     audio_ls = numpy()
    #     return audio_ls, labels_ls


class Speech_Commands(Dataset):
    """speech commands custom dataset"""

    def __init__(self, split="test", sample_len=25000):
        self.dataset = load_dataset(
            "speech_commands", "v0.01", split="test", keep_in_memory=True
        )
        self.ids = []
        for d in self.dataset:
            if d["speaker_id"] not in self.ids:
                self.ids.append(d["speaker_id"])
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

    def get_ids(self, s_id, tune=False):
        """get speaker ids for personlized speaker"""
        new_data = []
        s_id = self.ids[s_id]
        speaker_dict = {}
        for i, d in enumerate(self.dataset):
            speaker_dict[d["speaker_id"]] = 0
        if not s_id in self.ids:
            raise ValueError("incorrect ID please select from ", self.ids)
        for i, d in enumerate(self.dataset):
            speaker_dict[d["speaker_id"]] += 1
            if d["speaker_id"] == s_id:
                new_data.append(i)
        if tune:
            new_data = new_data[:5]
        return new_data

    def __getitem__(self, item):
        """return item from dataset"""
        audio = self.dataset[item]["audio"]["array"]
        label = self.dataset[item]["label"]
        # label = self.dataset[item]["file"].split("/")[0]
        sample_rate = 16000
        assert sample_rate == 16000
        mel = self.processor(
            audio, return_tensors="pt", sampling_rate=sample_rate
        ).input_values
        mel = mel.squeeze(0)
        # audio = self.processor(audio, padding="longest", return_tensors="pt")
        # audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        # mel = whisper.log_mel_spectrogram(audio)
        return mel, label


class LibriSpeech(Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """

    def __init__(self, split="test-clean", sample_len=20000):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )

        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

        self.max_len = sample_len

    def __len__(self):
        """get length of dataset"""
        return len(self.dataset)

    def __getitem__(self, item):
        """return audio item"""
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        audio = audio.squeeze(0)

        return audio, text

    def collate_calib_tokens(self, data):
        """collate calibration data"""
        input_audio = []
        label_list = []
        for d in data:
            inputs, labels = d
            mel = self.processor(
                inputs,
                return_tensors="pt",
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
                sampling_rate=16000,
            ).input_values
            input_audio.append(mel)
            label_list.append(labels)

        input_audio = torch.stack(input_audio, dim=0).squeeze(1).squeeze(1)

        return input_audio, label_list

    def collate_test_tokens(self, data):
        """collate test data"""
        data, labels = data[0]
        input_audio = self.processor(
            data, return_tensors="pt", sampling_rate=16000
        ).input_values
        # input_audio = []
        # processed_audio = []
        # label_list = []
        # for d in data:
        #     inputs, labels = d
        #     input_audio.append(inputs.squeeze(0))
        #     label_list.append(labels)
        # input_audio = torch.stack(input_audio, dim=0)
        return input_audio, labels

    def get_ids(self, s_id):
        """get specific speaker ids for personized task"""
        ids_set = []
        for i in range(len(self.dataset)):
            _, _, _, speaker, _, _ = self.dataset[i]
            if speaker == s_id:
                ids_set.append(i)
        return ids_set

    def get_genders(self, csv_path):
        """get gender subset"""
        m_subset = []
        f_subset = []
        print(csv_path)
        print(os.path.exists(csv_path))
        if os.path.exists(csv_path):
            df_data = pd.read_csv(
                csv_path, sep="|", skip_blank_lines=True, skipinitialspace=True
            )
        else:
            raise FileExistsError(
                "Missing CSV File for speakers. Ensure SPEAKERS.CSV is in /data/"
            )

        for i in range(len(self.dataset)):
            audio, sample_rate, text, speaker, _, _ = self.dataset[i]
            gender = df_data.loc[df_data["ID"] == speaker]["SEX"].values[0].strip()
            if gender == "M":
                m_subset.append(i)
            else:
                f_subset.append(i)
        return {"male": m_subset, "female": f_subset}


def get_data(args):
    """get dataset"""
    if args.tune_id and (
        not args.data == "libri" and not args.data == "speech_commands"
    ):
        raise NotImplementedError(
            "ID's only valid for librispeech - common_voice and image models not supported"
        )
    if args.data == "libri":
        dataset = LibriSpeech("test-clean", sample_len=args.tune_sample_len)
        if args.tune_gender or args.test_gender:
            gender_ids = dataset.get_genders(csv_p)
            if args.tune_gender:
                tune_ids = gender_ids[args.tune_gender]
                # tune_ids = tune_ids[:args.tune_samples] # select specific samples if required

                tune_dataset = Subset(dataset, tune_ids)
                tune_dataset, _ = random_split(
                    tune_dataset,
                    [args.tune_samples, len(tune_dataset) - args.tune_samples],
                )  # paper seed is 42

            if args.test_gender:
                test_ids = gender_ids[args.test_gender]
                test_dataset = Subset(dataset, test_ids)
                # test_dataset = Subset(dataset, gender_ids[args.test_gender])

        if args.tune_id or args.test_id:
            tune_ids = dataset.get_ids(args.tune_id)
            tune_ids = tune_ids[: args.tune_samples]
            tune_dataset = Subset(dataset, tune_ids)
            tune_dataset.max_len = args.tune_sample_len
            # if args.tune_id == args.test_id:
            #     test_dataset = tune_dataset
            # else:
            if args.test_id:
                if len(args.test_id) == 1:
                    args.test_id = args.test_id[0]
                test_ids = dataset.get_ids(args.test_id)
                test_dataset = Subset(dataset, test_ids)

        print("[+] Using dataset libri-clean-960h")
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            collate_fn=dataset.collate_test_tokens,
            shuffle=False,
        )
        tune_dataloader = DataLoader(
            tune_dataset,
            collate_fn=dataset.collate_calib_tokens,
            batch_size=args.bs,
            shuffle=False,
        )
        return tune_dataloader, test_dataloader

    elif args.data == "fleur":
        if args.tune_lang and args.test_lang:
            test_lang = args.test_lang

            test_dataset = Fleurs(test_lang, tune=False)
            tune_dataset = Fleurs(
                args.tune_lang, sample_len=args.tune_sample_len, tune=True
            )

            tune_dataset, _ = random_split(
                tune_dataset, [args.tune_samples, len(tune_dataset) - args.tune_samples]
            )

            tune_dataloader = DataLoader(tune_dataset, batch_size=1, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            return tune_dataloader, test_dataloader

    elif args.data == "speech_commands":
        dataset = Speech_Commands()
        tune_data = []

        if args.tune_id:
            tune_data = dataset.get_ids(args.tune_id, tune=True)
            tune_data = Subset(dataset, tune_data)

        if args.test_id:
            if type(args.test_id) == list:
                test_id = args.test_id[0]
            else:
                test_id = args.test_id
            test_data = dataset.get_ids(test_id, tune=False)
            test_data = Subset(dataset, test_data)

        tune_dataloader = DataLoader(tune_data, batch_size=args.bs, shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=args.bs, shuffle=False)

        return tune_dataloader, test_dataloader

    elif args.data == "common_voice":
        if args.tune_lang and args.test_lang:
            tune_data_loader = load_dataset(
                "common_voice",
                args.tune_lang,
                split=f"test[:{args.tune_samples}]",
                keep_in_memory=True,
            )

            test_data_loader = load_dataset(
                "common_voice",
                args.test_lang,
                split=f"test[:{args.test_samples}]",
                keep_in_memory=True,
            )
            print(
                f"[+] Using dataset common voice with tune:{args.tune_lang} test:{args.test_lang}"
            )
            tune_data_loader = common_voice_dataloader(
                data_filter(args, tune_data_loader, "len")
            )
            dataloader = tune_data_loader

            return None, None, dataloader, None
        else:
            return None, None, dataloader, None
    else:
        print(f"[!] Error: {args.data}: Dataset misconfiguration")
        raise NotImplementedError(
            "Dataset misconfiguration, dataset not recognized.\
                [common_voice, libri, cifar100, cifar10, imgnet]"
        )


def data_filter(args, dataset, data_filter="None"):
    """filter data based on args"""
    if data_filter == "id":
        return id_split(args, dataset)
    if data_filter == "gender":
        # returns male, female if both, else male OR female
        return get_gender_split(args, dataset)
    if data_filter == "cluster":
        pass
    elif data_filter == "age":
        pass
    elif data_filter == "len":
        print("filtering longest samples")
        data = dataset.filter(lambda d: d["audio"]["array"].shape[-1] > 100000)
        return data


def tidy_text(batch):
    """clean returned text"""
    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:"]'  # noqa: W605
    # speech, _ = torchaudio.load(batch["path"])
    # batch["speech"] = resampler.forward(speech.squeeze(0)).numpy()
    # batch["sampling_rate"] = resampler.new_freq
    batch["text"] = (
        re.sub(chars_to_ignore_regex, "", batch["text"]).upper().replace("’", "'")
    )
    return batch


def get_gender_split(args, test_dataloader):
    """split data based on speker ids"""
    csv_p = "./utils/SPEAKERS.csv"
    df_data = pd.read_csv(csv_p, sep="|", skip_blank_lines=True, skipinitialspace=True)
    genders = []
    for data in test_dataloader:
        speaker_id = data["speaker_id"]
        gender = df_data.loc[df_data["ID"] == speaker_id]["SEX"].values[0].strip()
        genders.append(gender)

    hg_data = test_dataloader.add_column("gender", genders)
    male_data = hg_data.filter(lambda example: example["gender"] == "M")
    female_data = hg_data.filter(lambda example: example["gender"] == "F")
    if args.gender == "male":
        test_dataloader = male_data
        print("[+] using Male data partition")
    elif args.gender == "female":
        test_dataloader = female_data
        print("[+] Using female data partition")
    elif args.tune_gender or args.test_gender:
        if args.tune_gender == "female":
            tune_dataloader = female_data
        elif args.tune_gender == "male":
            tune_dataloader = male_data
        if args.test_gender == "female":
            test_dataloader = female_data
        elif args.test_gender == "male":
            test_dataloader = male_data
        return tune_dataloader, test_dataloader
    else:
        return male_data, female_data
    return test_dataloader


def id_split(args, test_dataloader, id_t="all"):
    """filter based on id"""
    if id_t == "all":
        raise NotImplementedError
    data_loader = test_dataloader.filter(lambda d: d["speaker_id"] == id_t)
    return data_loader


def common_voice_dataloader(data):
    """load common voice data loader"""
    data = data.remove_columns(
        [
            "accent",
            "age",
            "client_id",
            "down_votes",
            "gender",
            "locale",
            "path",
            "segment",
            "up_votes",
        ]
    )
    data = data.cast_column("audio", Audio(sampling_rate=16000))

    return data


def to_torch_data(dataset, processor, max_len, batch_size):
    """map processor data to torch format"""
    if max_len:
        dataset = dataset.map(
            lambda e: processor(
                e["audio"][0]["array"],
                truncation=True,
                padding="max_length",
                max_length=max_len,
            ),
            batched=True,
        )
    else:
        dataset = dataset.map(
            lambda e: processor(
                e["audio"][0]["array"], truncation=True, padding="max_length"
            ),
            batched=True,
        )
    # dataset.set_format(type='torch', columns=["audio", "text"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size)
    return dataloader
