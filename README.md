# Minority-Focused Text-to-Image Generation via Prompt Optimization

This repository contains the code for the paper "Minority-Focused Text-to-Image Generation via Prompt Optimization".

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