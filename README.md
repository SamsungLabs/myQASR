<!-- Copyright (c) 2023 Samsung Electronics Co., Ltd.

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
For conditions of distribution and use, see the accompanying LICENSE.md file. -->

**A Model for Every User and Budget:**  **Label-Free and Personalized Mixed-Precision Quantization** 

<div align="center">

[![Paper](https://img.shields.io/badge/paper-INTERSPEECH2023-brightgreen)](https://arxiv.org/abs/2307.12659)
[![poster](https://img.shields.io/badge/poster-orange)](results/figures/poster_2023_IS.pdf)
[![BibTeX](https://img.shields.io/badge/Cite_us-BibTeX-blue)](#Citation)
 
 </div>

---
*Edward Fish (edward.fish@surrey.ac.uk), Umberto Michieli (u.michieli@samsung.com), Mete Ozay (m.ozay@samsung.com)*  
*Samsung Research UK, University of Surrey*  
Interspeech 2023

## Introduction

MyQASR is a mixed-precision quantization method that creates tailored quantization schemes for diverse users, all while meeting any memory requirement and without the need for fine-tuning. To do so, myQASR evaluates the quantization sensitivity of network layers by analyzing full-precision activation values given just a few unlabelled samples from the user. This simple but effective approach can generate a personalized mixed-precision quantization scheme that fits any predetermined memory budget with minimal performance degradation.

**Key Takeaways:**
- We show how different speakers require different mixed-precision quantization setups to preserve model performance.
- Since in most applications we have anonymized data and no labels we explore the effect of different speaker profiles on quantization sensitivity by passing unlabelled data though the network and observing activation statistics.
- Our empirical observation is that the mean activation values are a reasonable indicator of quantization sensitivity and so we design a method which generates a mixed precision setup based on this data.
- Since finding an optimal mixed precision setup for any size memory budget is intractable and large jumps in quantization resolutions between layers harms model performance, we introduce a uniformity constraint when selecting bit depths for each layer. 
- We show in the paper how this method works well for quantizing large ASR networks for different speaker profiles, including gender and language evaluated on [LibriSpeech](https://paperswithcode.com/dataset/librispeech), [Common Voice](https://commonvoice.mozilla.org/en), [Google Speech Commands](https://blog.research.google/2017/08/launching-speech-commands-dataset.html?m=1), and [FLEURS](https://arxiv.org/abs/2205.12446).

## Code Overview

We use HuggingFace Transformers for all models and the datasets are downloaded at runtime and partitioned into gender and language splits dynamically depending on the experiment you would like to run so there is no data or model pre-processing required! The code can be understood as follows: 

- Data preparation and separation:
	- Data is downloaded via the dataloader and partitioned into tune and test segments depending on the arguments provided. 
	
- Model Wrapping :
	- A pre-trained model is downloaded via huggingface. We use the relevant configuration file in `/configs/` to select which layers in the network will be quantized and they are initially set to 32 bit (full precision).  This takes place in the line: 
	
		``wrapped_modules, bit_depths = net_wrap.wrap_modules_in_net(
        model, quant_cfg)``
     - `wrapped_modules` is now a module dictionary containing the new quantizable modules with the methods `calibration_step` 
```py
    def forward(self, x):
	    # Layer is initialized as raw for sensitivity analysis
        if self.mode == 'raw':
            out = F.conv1d(x, self.weight, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)
        # optional mode for distance based metrics - will quantize layer temporarily to 8 bits.
        elif self.mode == "sensitivity_step":
	        out = self.sensitivity_step(x)
        # Will store raw layer inputs and gradients 
        elif self.mode == "calibration_step1":
            out = self.calibration_step1(x)
        # Perform quantization calibration in step 2
        elif self.mode == "calibration_step2":
            out = self.calibration_step2(x)
        else:
            raise NotImplementedError
        return out
   ```

- Sensitivity Analysis:
	- Sensitivity analysis is performed by collecting activation statistics via forward hooks while the layer is in mode `self.mode="raw"`. If a distance based metric is used then the layer will be switched to `self.mode="sensitivity_step"` which will quantize the layer, compute loss between both raw and quantized layers before resetting the layer to `mode="raw"`
- Quant Calibration:
   - Once  a bit configuration has been set, we simply update the operators with their relevant bit depths using `module.w_bit` and `module.w_qmax` . The `calibration_step=2` for the following stages. 
   - Hessian aware and cosine aware quantization is then performed to find optimum activation values for the layer using the current weight quantization bits and activation bits defined in the configuration file for the model. 
- Inference:
	- The model weights and activations are updated in-place during the `quant_calibrator.batching_quant_calib()` call. This function returns a new module_dict with the activation and weight bits and parameter count. It also includes additional info for logging. All results are logged under `results/` as csv files. 

- Visualisation:
	- A jupyter notebook is included to create the same figures as in the paper under `/utils/create_figs.ipynb` - 


## Adding new quantization or sensitivity methods
### Quantization sensitivity analysis
If you are interested in trying out new sensitivity methods you can take a look at the code in `utils/quant_calib_orig.py` here you will find the sensitivity methods which compare the raw 32 bit activations with the quantized activations:

```py
	elif args.sensitivity_method == "dist_l1":
		dist = -torch.abs(raw_preds_flat - q_preds)
		dist = torch.abs(torch.mean(dist, dim=-1)).cpu()
 ```
 
 The `raw_preds_flat` are the concatenated output values for that layer and the `q_preds` are the quantized ones. It should be simple to add your own method here and then call it via the arguments. 
### Quantization calibration
To add additional calibration methods for quantization you can add a different calibration method in `/quant_layers`. You will want to update the code for each module to use an alternative to `calibration_step2`. For example:

```py
def calibration_step2(self, x):
	    # initialize intervals with minmax intervals
	    self._initialize_intervals(x)

        # put raw outs on GPU
        self.raw_out = self.raw_out.to(
            x.device).unsqueeze(1)  # shape: B,1,oc,W,H

        # put raw grad on GPU
        self.raw_grad = self.raw_grad.to(
            x.device) if self.raw_grad != None else None

		# YOUR NEW QUANTIZATION CALIBRATION SEARCH METHOD HERE!
		
        self.raw_grad = self.raw_grad.to(
            "cpu") if self.raw_grad != None else None
		
		# ensure you set the layer to calibrated or there will be issues.
        self.calibrated = True
        out = self.quant_forward(x)
        del self.raw_input, self.raw_out, self.raw_grad
        return out
   ```
## Installation
 First install all necessary libraries.
```bash
pip install -r requirements.txt
```
Our setup used:
- `python==3.9`
- `torch==2.0.1`
- `torch-audio==2.0.2`
- `cuda==11.7`

## Usage
You can mix and match models, datasets, and quantization methods.

To quantize a **wav2vec-base model** to 75 mb with cosine calibration, median sensitivity metric, and male tuning and testing partition of **librispeech** (Table 3 in our results):

```bash
python3 main.py --model wav2vec-base --bs 32 --experiment reproduce --data libri --tune_gender male --test_gender male --sensitivity_method median --calibration_method cosine --target 75
```

We provide the script ```run_gender.sh``` to replicate the results from Table 3 of our main paper. Note: the final results will be slightly better than those reported in the paper thanks to additional optimization to the sample lenght parameter during refactoring.

To run the same experiment but with a librispeech **user ID** instead of a gender partition you can swap the `--tune_gender` and `--test_gender` for `--tune_id` and `--test_id`. 

**For example:**

`python3 main.py --model wav2vec-base --bs 32 --experiment git_test --data libri --
tune_id 121 --test_id 121 --calibration_method cosine --target 75`

To run experiments on Wav2Vec-Conformer with Google Speech Commands with tune and testing of user id you can change `--data` and `--model`

```python3 main.py --model wav2vec-conformer --bs 32 --experiment git_test --data speech_commands --tune_id 1 --test_id 1 --sensitivity_method median --calibration_method cosine --target 2000```

<a name="Citation"></a>
## Citation

Please cite our work if you use this codebase:

```
@article{fish2023model,
  title={A Model for Every User and Budget: Label-Free and Personalized Mixed-Precision Quantization},
  author={Fish, Edward and Michieli, Umberto and Ozay, Mete},
  journal={INTERSPEECH},
  year={2023}
}
```

The hessian calibration method is adapted from [PTQ4ViT](https://github.com/hahnyuan/PTQ4ViT). Please also cite their work if you use this codebase:

```
@article{PTQ4ViT,
    title={PTQ4ViT: Post-Training Quantization Framework for Vision Transformers},
    author={Zhihang Yuan, Chenhao Xue, Yiqi Chen, Qiang Wu, Guangyu Sun},
    journal={European Conference on Computer Vision (ECCV)},
    year={2022},
}
```