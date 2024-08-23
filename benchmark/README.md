Official Pytorch implementation of 3D Net, from the following paper:

We propose **3D Net**, a pure volumetric convolutional network to adapt hierarchical transformers behaviour (e.g. Swin Transformer) for Medical Image Segmentation with less model parameters.

 ## Installation
 Please look into the [INSTALL.md](INSTALL.md) for creating conda environment and package installation procedures.

 (Feel free to post suggestions in issues of recommending latest proposed transformer network for comparison. Currently, the network folder is to put the current SOTA transformer. We can further add the recommended network in it for training.)
 
 <!-- ✅ ⬜️  -->
 
 ## Results 
 | Methods | resolution | #params | FLOPs | Mean Dice (AMOS2022) |
|:---:|:---:|:---:|:---:| :---:|
| TransBTS | 96x96x96 | 31.6M | 110.4G | 0.792 |
| UNETR | 96x96x96 | 92.8M | 82.6G | 0.762 | 
| nnFormer | 96x96x96 | 149.3M | 240.2G | 0.790 | 
| SwinUNETR | 96x96x96 | 62.2M | 328.4G | 0.880 | 
| 3D UX-Net | 96x96x96 | 53.0M | 639.4G | 0.900 (kernel=7) |

<!-- ✅ ⬜️  -->
## Training
Training and fine-tuning instructions are in [TRAINING.md](TRAINING.md). Pretrained model weights will be uploaded for public usage later on.

<!-- ✅ ⬜️  -->
## Evaluation
Efficient evaulation can be performed for the above three public datasets as follows:
```

```

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```
@article{,
}
```

 
 


