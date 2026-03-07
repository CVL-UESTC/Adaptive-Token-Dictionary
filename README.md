# Adaptive Token Dictionary

This repository is an official implementation of the papers ["Transcending the Limit of Local Window: Advanced Super-Resolution Transformer with Adaptive Token Dictionary"](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Transcending_the_Limit_of_Local_Window_Advanced_Super-Resolution_Transformer_with_CVPR_2024_paper.html) (CVPR 2024), and ["ATD: Improved Transformer with Adaptive Token Dictionary for Image Restoration"](https://www.computer.org/csdl/journal/tp/5555/01/11419871/2eyKyLHhok0) (extended version accepted by TPAMI)

[![arXiv](https://img.shields.io/badge/arXiv-2401.08209-b31b1b.svg)](https://arxiv.org/abs/2401.08209)
[![arXiv](https://img.shields.io/badge/arXiv-2603.02581-b31b1b.svg)](https://arxiv.org/abs/2603.02581)
[![Visual Results](https://img.shields.io/badge/Visual%20Results-4285F4?logo=google-drive&logoColor=white)](https://drive.google.com/drive/folders/16Jy7MR6n7kRUjfX7tRJlz_20f3Y-kZhc?usp=sharing)
[![Visual Results](https://img.shields.io/badge/Pretrained%20Models-4285F4?logo=google-drive&logoColor=white)](https://drive.google.com/drive/folders/1nz0I798GM8HkPYVZCLe8i9LAK31mpYQs?usp=sharing)

By [Leheng Zhang](https://scholar.google.com/citations?user=DH1CJqkAAAAJ), [Wei Long](https://scholar.google.com/citations?user=CsVTBJoAAAAJ), [Yawei Li](https://scholar.google.com/citations?user=IFLsTGsAAAAJ), [Xingyu Zhou](https://scholar.google.com/citations?user=dgO3CyMAAAAJ), [Xiaorui Zhao](https://scholar.google.com/citations?user=kiG_knoAAAAJ), and [Shuhang Gu](https://scholar.google.com/citations?user=-kSTt40AAAAJ).

> **Abstract:** Recently, Transformers have gained significant popularity in image restoration tasks such as image super-resolution and denoising, owing to their superior performance. However, balancing performance and computational burden remains a long-standing problem for transformer-based architectures. Due to the quadratic complexity of self-attention, existing methods often restrict attention to local windows, resulting in limited receptive field and suboptimal performance. To address this issue, we propose Adaptive Token Dictionary (ATD), a novel transformer-based architecture for image restoration that enables global dependency modeling with linear complexity relative to image size. The ATD model incorporates a learnable token dictionary, which summarizes external image priors (i.e., typical image structures) during the training process. To utilize this information, we introduce a token dictionary cross-attention (TDCA) mechanism that enhances the input features via interaction with the learned dictionary. Furthermore, we exploit the category information embedded in the TDCA attention maps to group input features into multiple categories, each representing a cluster of similar features across the image and serving as an attention group. We also integrate the learned category information into the feed-forward network to further improve feature fusion. ATD and its lightweight version ATD-light, achieve state-of-the-art performance on multiple image super-resolution benchmarks. Moreover, we develop ATD-U, a multi-scale variant of ATD, to address other image restoration tasks, including image denoising and JPEG compression artifacts removal. Extensive experiments demonstrate the superiority of out proposed models, both quantitatively and qualitatively. 
> 
> <img width="800" src="figures/attention.png"> 
> <br/><br/>
> <img width="800" src="figures/model.png"> 
> <br/><br/>
> <img width="800" src="figures/arch.png">



## Contents›
1. [Enviroment](#environment)
1. [Fast Inference for SR](#fast-inference-for-sr)
1. [Training Instruction](#training-instruction)
1. [Testing Instruction](#testing)
1. [Results](#results)
1. [Visual Results](#visual-results)
1. [Citation](#citation)
1. [Acknowledgements](#acknowledgements)


## Environment
- Python 3.11
- PyTorch 2.7.0

### Installation
```bash
git clone https://github.com/LabShuHangGU/Adaptive-Token-Dictionary.git

conda create -n ATDv2 python=3.11
conda activate ATDv2

pip install -r requirements.txt
pip install git+https://github.com/KellerJordan/Muon
python setup.py develop
```

## Fast Inference for SR
Using ```inference.py``` for fast inference on single image or multiple images within the same folder.
```bash
# For classical SR
python inference.py -i test_image.png -o results/test/ --scale 4 --task classical
python inference.py -i test_images/ -o results/test/ --scale 4 --task classical

# For lightweight SR
python inference.py -i test_image.png -o results/test/ --scale 4 --task lightweight
python inference.py -i test_images/ -o results/test/ --scale 4 --task lightweight
```
The ATD SR model processes the image ```test_image.png``` or images within the ```test_images/``` directory. The results will be saved in the ```results/test/``` directory.


## Training Instruction
### Datasets
Used training and testing sets can be downloaded as follows:
| Task | Training Set | Testing Set |
| :--- | :---: | :---: |
| Classical Image SR | [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (800 images) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) | Set5 + Set14 + B100 + Urban100 + Mange109 <br> [\[download\]](https://drive.google.com/file/d/1_FvS_bnSZvJWx9q4fNZTR8aS15Rb0Kc6/view?usp=sharing) |
| Lightweight Image SR | [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (800 images) | Set5 + Set14 + B100 + Urban100 + Mange109 <br> [\[download\]](https://drive.google.com/file/d/1_FvS_bnSZvJWx9q4fNZTR8aS15Rb0Kc6/view?usp=sharing) |
| Color Gaussian Image Denoising | [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (800 images) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) + [WED](https://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar) (4744 images) + [BSD500](https://drive.google.com/file/d/12oYmda-cC54XG5019EI17g0GG9qxybe9/view?usp=sharing) (500 images) | CBSD68 + Kodak24 + McMaster + Urban100 <br> [\[download\]](https://drive.google.com/file/d/11374-202z7ULaA50FKUMXAlVeOykTvGo/view?usp=sharing) |
| Grayscale Gaussian Image Denoising | [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (800 images) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) + [WED](https://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar) (4744 images) + [BSD500](https://drive.google.com/file/d/12oYmda-cC54XG5019EI17g0GG9qxybe9/view?usp=sharing) (500 images) | Set12 + BSD68 + Urban100 <br> [\[download\]](https://drive.google.com/file/d/1zFESymnrJn39sSkcTu8XbMByDEy8_yse/view?usp=sharing) |
| Grayscale JPEG CAR | [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (800 images) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) + [WED](https://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar) (4744 images) + [BSD500](https://drive.google.com/file/d/12oYmda-cC54XG5019EI17g0GG9qxybe9/view?usp=sharing) (500 images) | Classic5 + LIVE1 + Urban100 <br> [\[download\]](https://drive.google.com/file/d/16rZLCcLg9yXD9jyromwy4UL0A5AhU_Wm/view?usp=sharing) |

After downloading the datasets and putting them into the folder `datasets/`, use the code `python scripts/extract_subimages.py` and `python scripts/generate_patches_dfwb.py` to extract image patches for training.

### Training Commands
Refer to the training configuration files in `options/train` folder for detailed settings.

<details>
<summary>ATD (Classical Image Super-Resolution)</summary>

```bash
# batch size = 4 (GPUs) × 12 (per GPU)
# training dataset: DF2K

# ×2 scratch, input size = 64×64, 300k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/sr/000_ATD_SRx2_scratch.yml --launcher pytorch
# ×2 finetune, input size = 96×96, 250k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/sr/001_ATD_SRx2_finetune.yml --launcher pytorch

# ×3 finetune, input size = 96×96, 250k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/sr/002_ATD_SRx3_finetune.yml --launcher pytorch

# ×4 finetune, input size = 96×96, 250k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/sr/003_ATD_SRx4_finetune.yml --launcher pytorch
```
</details>

<details>
<summary>ATD-light (Lightweight Image Super-Resolution)</summary>

```bash
# batch size = 2 (GPUs) × 32 (per GPU)
# training dataset: DIV2K

# ×2 scratch, input size = 64×64, 500k iterations
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --use-env --nproc_per_node=2 --master_port=1145  basicsr/train.py -opt options/train/sr/101_ATD_light_SRx2_scratch.yml --launcher pytorch

# ×3 finetune, input size = 64×64, 250k iterations
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --use-env --nproc_per_node=2 --master_port=1145  basicsr/train.py -opt options/train/sr/102_ATD_light_SRx3_finetune.yml --launcher pytorch

# ×4 finetune, input size = 64×64, 250k iterations
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --use-env --nproc_per_node=2 --master_port=1145  basicsr/train.py -opt options/train/sr/103_ATD_light_SRx4_finetune.yml --launcher pytorch
```
</details>

<details>
<summary>ATD-U (Color Gaussian Image Denoising)</summary>

```bash
# training dataset: DFWB

# sigma=15, input size = 24x256x256 -> 12x384x384, 100k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/denoising/201_ATD_U_denoising_color_sigma15.yml --launcher pytorch

# sigma=25, input size = 24x256x256 -> 12x384x384, 100k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/denoising/202_ATD_U_denoising_color_sigma25.yml --launcher pytorch

# sigma=50, input size = 24x256x256 -> 12x384x384, 100k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/denoising/203_ATD_U_denoising_color_sigma50.yml --launcher pytorch
```
</details>

<details>
<summary>ATD-U (Grayscale Gaussian Image Denoising)
</summary>

```bash
# sigma=15, input size = 24x256x256 -> 12x384x384, 100k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/denoising/301_ATD_U_denoising_gray_sigma15.yml --launcher pytorch

# sigma=25, input size = 24x256x256 -> 12x384x384, 100k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/denoising/302_ATD_U_denoising_gray_sigma25.yml --launcher pytorch

# sigma=50, input size = 24x256x256 -> 12x384x384, 100k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/denoising/303_ATD_U_denoising_gray_sigma50.yml --launcher pytorch
```
</details>

<details>
<summary>ATD-U (Grayscale JPEG CAR)</summary>

```bash
# q=10, input size = 24x256x256 -> 12x384x384, 100k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/jpegcar/401_ATD_U_jpegcar_gray_q10.yml --launcher pytorch

# q=20, input size = 24x256x256 -> 12x384x384, 100k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/jpegcar/402_ATD_U_jpegcar_gray_q20.yml --launcher pytorch

# q=30, input size = 24x256x256 -> 12x384x384, 100k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/jpegcar/403_ATD_U_jpegcar_gray_q30.yml --launcher pytorch

# q=40, input size = 24x256x256 -> 12x384x384, 100k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/jpegcar/404_ATD_U_jpegcar_gray_q40.yml --launcher pytorch
```

</details>

## Testing
### Data Preparation
Download the testing data and put them in the folder `datasets/`.

### Pretrained Models
Download the [pretrained models](https://drive.google.com/drive/folders/1nz0I798GM8HkPYVZCLe8i9LAK31mpYQs?usp=sharing) and put them in the folder `experiments/pretrained_models/`.

### Testing Commands
Refer to the testing configuration files in `options/test/` folder for detailed settings.

<details>
<summary>ATD (Classical Image Super-Resolution)
</summary>

```bash
python basicsr/test.py -opt options/test/001_ATD_SRx2_finetune.yml
python basicsr/test.py -opt options/test/002_ATD_SRx3_finetune.yml
python basicsr/test.py -opt options/test/003_ATD_SRx4_finetune.yml
```

</details>

<details>
<summary>ATD-light (Lightweight Image Super-Resolution)
</summary>

```bash
python basicsr/test.py -opt options/test/101_ATD_light_SRx2_scratch.yml
python basicsr/test.py -opt options/test/102_ATD_light_SRx3_finetune.yml
python basicsr/test.py -opt options/test/103_ATD_light_SRx4_finetune.yml
```

</details>

<details>
<summary>ATD-U (Color Gaussian Image Denoising)
</summary>

```bash
python basicsr/test.py -opt options/test/denoising/201_ATD_U_denoising_color_sigma15.yml
python basicsr/test.py -opt options/test/denoising/202_ATD_U_denoising_color_sigma25.yml
python basicsr/test.py -opt options/test/denoising/203_ATD_U_denoising_color_sigma50.yml
```

</details>

<details>
<summary>ATD-U (Grayscale Gaussian Image Denoising)
</summary>

```bash
python basicsr/test.py -opt options/test/denoising/301_ATD_U_denoising_gray_sigma15.yml
python basicsr/test.py -opt options/test/denoising/302_ATD_U_denoising_gray_sigma25.yml
python basicsr/test.py -opt options/test/denoising/303_ATD_U_denoising_gray_sigma50.yml
```

</details>

<details>
<summary>ATD-U (Grayscale JPEG CAR)
</summary>

```bash
python basicsr/test.py -opt options/test/jpegcar/401_ATD_U_jpegcar_gray_q10.yml
python basicsr/test.py -opt options/test/jpegcar/402_ATD_U_jpegcar_gray_q20.yml
python basicsr/test.py -opt options/test/jpegcar/403_ATD_U_jpegcar_gray_q30.yml
python basicsr/test.py -opt options/test/jpegcar/404_ATD_U_jpegcar_gray_q40.yml
```

</details>

## Results

<details>
<summary>ATD (Classical Image Super-Resolution)</summary>

<img width="800" src="figures/results_sr_classical.png">

</details>

<details>
<summary>ATD-light (Lightweight Image Super-Resolution)</summary>

<img width="800" src="figures/results_sr_lightweight.png">

</details>

<details>
<summary>ATD-U (Gaussian Image Denoising)</summary>

<img width="800" src="figures/results_denoising.png">

</details>

<details>
<summary>ATD-U (Grayscale JPEG CAR)</summary>

<img width="800" src="figures/results_jpegcar.png">

</details>

## Visual Results

Complete visual results can be downloaded from [link](https://drive.google.com/drive/folders/1HwEbAGU6WEw9ZGbFdt__BOJo_5DKflEb?usp=sharing).

<details>
<summary>ATD (Classical Image Super-Resolution)</summary>

<img width="800" src="figures/visual_sr.png">

</details>

<details>
<summary>ATD-U (Gaussian Image Denoising)</summary>

<img width="800" src="figures/visual_denoising.png">

</details>

<details>
<summary>ATD-U (Grayscale JPEG CAR)</summary>

<img width="800" src="figures/visual_jpegcar.png">

</details>


## Citation

```
@InProceedings{Zhang_2024_CVPR,
    author    = {Zhang, Leheng and Li, Yawei and Zhou, Xingyu and Zhao, Xiaorui and Gu, Shuhang},
    title     = {Transcending the Limit of Local Window: Advanced Super-Resolution Transformer with Adaptive Token Dictionary},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {2856-2865}
}

@article{zhang2026atd,
    title={ATD: Improved Transformer With Adaptive Token Dictionary for Image Restoration},
    author={Zhang, Leheng and Long, Wei and Li, Yawei and Zhou, Xingyu and Zhao, Xiaorui and Gu, Shuhang},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year={2026},
    publisher={IEEE}
}
```

## Acknowledgements
This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR).

