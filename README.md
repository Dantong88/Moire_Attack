# Moiré Attack (MA): A New Potential Risk of Screen Photos [NeurIPS 2021]

This repository is the official implementation of [Moiré Attack (MA): A New Potential Risk of Screen Photos](https://arxiv.org/abs/2030.12345). 

<img src="Images/Pipeline.png" alt="image" style="zoom:20%;" />

Images, captured by a camera, play a critical role in training Deep Neural Networks (DNNs). Usually, we assume the images acquired by cameras are consistent with the ones perceived by human eyes. However, due to the different physical mechanisms between human-vision and computer-vision systems, the final perceived images could be very different in some cases, for example shooting on digital monitors. In this work, we find a special phenomenon in digital image processing, the moiré effect, that could cause unnoticed security threats to DNNs. Based on it, we propose a Moiré Attack (MA) that generates the physical-world moiré pattern adding to the images by mimicking the shooting process of digital devices. Extensive experiments demonstrate that our proposed digital Moiré Attack (MA) is a perfect camouflage for attackers to tamper with DNNs with a high success rate (100.0% for untargeted and 97.0% for targeted attack when the noise budget $\epsilon=4$), high transferability rate across different models, and high robustness under various defenses. Furthermore, MA is with great stealthiness because the moiré effect is unavoidable due to the camera's inner physical structure, which therefore hardly attracts the awareness of humans.

## Requirements
We need torch >= 1.4, torchattacks = 2.12.2, colour-demosaicing = 0.1.6

To install requirements:

```setup
pip install [Package]
```

## Create an environment

```
conda create -n Moire_Attack_env python=3.7
source activate Moire_Attack_env
```

## Quick Start

```
python Moire_Attack.py
```

## Acknowledgments
* We referred [RjDuan's code' style](https://github.com/RjDuan/AdvDrop)
* We referred [Harry's code style] (https://github.com/Harry24k/adversarial-attacks-pytorch)

## Citation
```

```
