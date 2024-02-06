: '
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Edward Fish (edward.fish@samsung.com; edward.fish@surrey.ac.uk)
Umberto Michieli (u.michieli@samsung.com)
Mete Ozay (m.ozay@samsung.com)

Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.
'

# This script replicates our results on the gender personalization case (see Table 3 of the main paper).


# MALE-MALE case

export CUDA_VISIBLE_DEVICES=0
python3 main.py --model wav2vec-base --bs 1 --tune_samples 100 --experiment reproduce --data libri --tune_gender male --test_gender male --sensitivity_method median --calibration_method cosine --target 75 &

# export CUDA_VISIBLE_DEVICES=1
# python3 main.py --model wav2vec-base --bs 1 --tune_samples 100 --experiment reproduce --data libri --tune_gender male --test_gender male --sensitivity_method median --calibration_method hessian --target 75 &



# FEMALE-FEMALE case

export CUDA_VISIBLE_DEVICES=2
python3 main.py --model wav2vec-base --bs 1 --tune_samples 100 --experiment reproduce --data libri --tune_gender female --test_gender female --sensitivity_method median --calibration_method cosine --target 75 &

# export CUDA_VISIBLE_DEVICES=3
# python3 main.py --model wav2vec-base --bs 1 --tune_samples 100 --tune_sample_len 10000 --experiment reproduce --data libri --tune_gender female --test_gender female --sensitivity_method median --calibration_method hessian --target 75 &

