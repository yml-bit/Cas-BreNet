#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os

"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

# export nnUNet_raw="/home/wangtao/yml/project/python39/p3/nnUNet/DATASET/nnUNet_raw"
# export nnUNet_preprocessed="/home/wangtao/yml/project/python39/p3/nnUNet/DATASET/nnUNet_preprocessed"
# export nnUNet_results="/home/wangtao/yml/project/python39/p3/nnUNet/DATASET/nnUNet_results_unet"

nnUNet_raw = '/root/autodl-tmp/yml/project/p3/nnUNet/DATASET/nnUNet_raw'
nnUNet_preprocessed = '/root/autodl-tmp/yml/project/p3/nnUNet/DATASET/nnUNet_preprocessed'
nnUNet_results = '/root/autodl-tmp/yml/project/p3/nnUNet/DATASET/nnUNet_results_unet'#m:medium n:no supervision nnUNet_results_MedNeXt_v2

# nnUNet_raw = '/root/autodl-tmp/yml/project/p3/nnUNet/DATASET1/nnUNet_raw'
# nnUNet_preprocessed = '/root/autodl-tmp/yml/project/p3/nnUNet/DATASET1/nnUNet_preprocessed'
# nnUNet_results = '/root/autodl-tmp/yml/project/p3/nnUNet/DATASET1/nnUNet_results_MedNeXt_v2'#m:medium n:no supervision nnUNet_results_MedNeXt_v2  breast

if nnUNet_raw is None:
    print("nnUNet_raw is not defined and nnU-Net can only be used on data for which preprocessed files "
          "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set "
          "this up properly.")

if nnUNet_preprocessed is None:
    print("nnUNet_preprocessed is not defined and nnU-Net can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how "
          "to set this up.")

if nnUNet_results is None:
    print("nnUNet_results is not defined and nnU-Net cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information "
          "on how to set this up.")
