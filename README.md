# Minority-Focused Text-to-Image Generation via Prompt Optimization (CVPR 2025)

Soobin Um and [Jong Chul Ye](https://bispl.weebly.com/professor.html)

This repository contains the code for the paper "Minority-Focused Text-to-Image Generation via Prompt Optimization" (CVPR 2025).

## Setup

First, create your environment. We recommand to use the following comments. 

```
git clone https://github.com/anonymous5293/MinorityPrompt
cd MinorityPrompt
conda env create -f environment.yaml
```

Second, download [sdxl_lightning_4step_unet.safetensors](https://huggingface.co/ByteDance/SDXL-Lightning/tree/main) in ```ckpt```.


## Examples

- T2I generation
```
source scripts/text_to_img.sh
```

- MS-COCO
```
source scripts/text_to_mscoco.sh
```

Feel free to modify the scripts to fit your needs.

## Citation
If you find this repository useful, please cite our paper:
```
@article{um2024minorityprompt,
  title={MinorityPrompt: Text to Minority Image Generation via Prompt Optimization},
  author={Um, Soobin and Ye, Jong Chul},
  journal={arXiv preprint arXiv:2410.07838},
  year={2024}
}
```
