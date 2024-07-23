This is an implementation of my work on Class-Based Adaptive Instance Normalization. This work was done as an internship project at The Complex Systems Engineering Department (DISC) of ISAE-SUPAERO under the supervision of Dr. Thomas Oberlin. 

The code in this repository builds on the [Unofficial Pytorch Implementation of AdaIN](https://github.com/naoto0804/pytorch-AdaIN.git) by [Naoto Inoue](https://github.com/naoto0804). I am sincerely grateful for the unofficial implementation as well as the [Original Implementation in Torch](https://github.com/xunhuang1995/AdaIN-style.git) by the authors of Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization [Huang+, ICCV2017].

# Class-Based AdaIN
This work presents Class-Based Adaptive Instance Normalization (CB-AdaIN), as an approach to style transfer for domain adaptation in urban scene segmentation. We address limitations of standard AdaIN when applied to complex urban scenes by incorporating class-specific style statistics. Our method computes style statistics across multiple reference images for each semantic class, allowing for more contextually appropriate style transfer. We demonstrate CB-AdaIN's effectiveness using both a custom encoder-decoder architecture and a pre-trained Stable Diffusion VAE. Our results, show improved color adaptation and reduced artifacts compared to standard AdaIN. While the proposed method has visible limitations in generating realistic images, with improvements to the decoder architecture, CB-AdaIN shows potential to be used effectively for style transfer in urban scenes.

## Requirements
Please install requirements by `pip install -r requirements_pip.txt`. Alternatively, if you are using a conda environment, install requirements by `conda create --name <env> --file requirements_conda.txt`. 

This implementation uses Python 3.9 and Cuda 11.8. 

## Usage
### Download & Organize Data
1. Download the GTA5 and Cityscapes datasets from the official websites.
2. Store the test or train images under `input/content/GTA/images/test/` or `input/content/GTA/images/train/` and `input/style/Cityscapes/images/test/` or `input/style/Cityscapes/images/train/` respectively.
3. Store the RGB labels (segmentation maps) under `input/content/GTA/labels_original/` and `input/style/Cityscapes/labels_original/` respectively.
4. Run the [script](./utils/segmap_to_cityscapes_labelIds.py) as `python /utils/segmap_to_cityscapes_labelIds.py`. The grayscale images with Cityscapes label Ids will be saved under `input/content/GTA/labels/` and `input/style/Cityscapes/labels/` respectively.

### Download models


### Test
Use `--content` and `--style` to provide the respective path to the content and style image.
```
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content input/content/cornell.jpg --style input/style/woman_with_hat_matisse.jpg
```

You can also run the code on directories of content and style images using `--content_dir` and `--style_dir`. It will save every possible combination of content and styles to the output directory.
```
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content_dir input/content --style_dir input/style
```

Some other options:
* `--content_size`: New (minimum) size for the content image. Keeping the original size if set to 0.
* `--style_size`: New (minimum) size for the content image. Keeping the original size if set to 0.
* `--alpha`: Adjust the degree of stylization. It should be a value between 0.0 and 1.0 (default).
* `--preserve_color`: Preserve the color of the content image.

For more details and parameters, please refer to --help option.

### Train
Use `--content_dir` and `--style_dir` to provide the respective directory to the content and style images.
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py --content_dir <content_dir> --style_dir <style_dir>
```
Some other options:
* `--content_size`: New (minimum) size for the content image. Keeping the original size if set to 0.
* `--style_size`: New (minimum) size for the content image. Keeping the original size if set to 0.
* `--alpha`: Adjust the degree of stylization. It should be a value between 0.0 and 1.0 (default).
* `--preserve_color`: Preserve the color of the content image.

For more details and parameters, please refer to --help option.


## References
- [1]: [Unofficial Pytorch Implementation of AdaIN](https://github.com/naoto0804/pytorch-AdaIN.git) by [Naoto Inoue](https://github.com/naoto0804)
- [2]: X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", in ICCV, 2017.
- [3]: [Original implementation in Torch](https://github.com/xunhuang1995/AdaIN-style)
- [4]: M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele. "The Cityscapes Dataset for Semantic Urban Scene Understanding." in CVPR, 2016.
- [5]: S. R. Richter*, V. Vineet*, S. Roth, and V. Koltun. "Playing for Data: Ground Truth from Computer Games." in ECCV, 2016. (*equal contribution)
